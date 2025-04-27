import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from UniversalCUSUMDetector import UniversalCUSUMDetector

class ModelPipeline:
    def __init__(
        self,
        models_config: dict,
        n_splits: int = 3,
        feature_selector_top_k: int = 20,
        n_trials: int = 10,
        metric_func=mean_absolute_error,
        optuna_direction: str = "minimize",
        use_optuna: bool = True,
    ):
        self.models_config = models_config
        self.n_splits = n_splits
        self.top_k = feature_selector_top_k
        self.n_trials = n_trials
        self.metric_func = metric_func
        self.optuna_direction = optuna_direction
        self.use_optuna = use_optuna
        self.best_params = {}

    def tune_model(self, model_name, X, y, key_rate):
        def objective(trial):
            model_cls, param_space = self.models_config[model_name]
            params = {k: v(trial) for k, v in param_space.items()}
            tscv = TimeSeriesSplit(n_splits=self.n_splits)

            fold_losses = []

            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                rate_val = key_rate.iloc[val_idx]

                model = model_cls(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)

                fold_losses.append(self.metric_func(y_val, y_pred, rate_val))

            return np.mean(fold_losses)

        study = optuna.create_study(direction=self.optuna_direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        return study.best_params

    def run(self, X, y, key_rate, test_start_date, model_name, extra_metrics=None):
        if extra_metrics is None:
            extra_metrics = []

        results = []

        balance_reference = y[(y > 0) & (y.index >= test_start_date)].mean()

        X_train = X[X.index < test_start_date]
        X_test = X[X.index >= test_start_date]
        y_train = y.loc[X_train.index]
        y_test = y.loc[X_test.index]
        rate_train = key_rate.loc[X_train.index]
        rate_test = key_rate.loc[X_test.index]

        if model_name not in self.models_config:
            raise ValueError(f"Модель '{model_name}' не найдена в models_config.")

        if self.use_optuna:
            print(f"\n Калибровка модели: {model_name}")
            best_params = self.tune_model(model_name, X_train, y_train, rate_train)
        else:
            print(f"\n Обучение {model_name} без тюнинга (параметры по умолчанию)")
            best_params = {}  # Просто пустой словарь

        self.best_params[model_name] = best_params

        model_cls = self.models_config[model_name][0]
        selector_model = model_cls(**best_params)
        selector_model.fit(X_train, y_train)
        top_features = self._select_top_features(selector_model, X_train)
        print(f"Топ-{self.top_k} признаков для {model_name}: {top_features}")

        if X_test.empty:
            print("Нет тестовых данных. Обучение модели без теста.")
            model = model_cls(**best_params)
            model.fit(X_train[top_features], y_train)
            return None

        # делаем финальное обучение только по топовым признакам
        model = model_cls(**best_params, verbose=0)
        model.fit(X_train[top_features], y_train)
        y_pred = model.predict(X_test[top_features])

        results.append(
            self._evaluate(model_name, y_test, y_pred, rate_test, balance_reference, extra_metrics)
        )

        self._plot(y_test, y_pred, model_name)

        return pd.DataFrame(results)

    # остальное без изменений


    def _select_top_features(self, model, X):
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            raise ValueError("Model does not support feature importance or coef_.")
        
        top_indices = np.argsort(importances)[::-1][:self.top_k]
        return list(X.columns[top_indices])

    def _evaluate(self, name, y_true, y_pred, rate, balance_ref, extra_metrics):
        result = {
            "Model": name,
            "MAE": mean_absolute_error(y_true, y_pred),
            self.metric_func.__name__: self.metric_func(y_true, y_pred, rate),
        }
        for func in extra_metrics:
            result[func.__name__] = func(y_true, y_pred, rate)
        return result

    def _plot(self, y_true, y_pred, model_name):
        plt.figure(figsize=(15, 6))
        plt.plot(y_true.index, y_true.values, label="Истинное значение", linewidth=2)
        plt.plot(y_true.index, y_pred, label=f"Предсказание ({model_name})", linewidth=2)
        plt.title(f"Факт vs Прогноз для {model_name}")
        plt.xlabel("Дата")
        plt.ylabel("Сальдо")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def online_run(
        self,
        X,
        y,
        key_rate,
        start_date,
        detector: UniversalCUSUMDetector,
        retrain_every_n_sundays=2,
        reoptimize_every_m_sundays=4,
        model_name="LGBM",
        extra_metrics=None,
    ):
        if extra_metrics is None:
            extra_metrics = []
        
        self.daily_results = []
        self.daily_predictions = []
        self.change_points = []
        
        # --- Первичное обучение до start_date ---
        X_train = X[X.index < start_date]
        y_train = y[X.index < start_date]
        key_rate_train = key_rate[X.index < start_date]

        print(f"Первичное обучение до {start_date.date()}")
        self.use_optuna = True
        self._fit(X_train, y_train, key_rate_train, model_name)

        # --- Прогон в онлайн-режиме ---
        current_date = start_date
        sunday_counter = 0  # считаем воскресенья
        all_dates = X.index[X.index >= start_date]

        y_pred_total = []
        self.residuals = []


        while current_date <= all_dates[-1]:
            X_today = X.loc[[current_date]]
            y_today = y.loc[[current_date]]
            key_rate_today = key_rate.loc[[current_date]]
            
            # --- Предсказание ---
            y_pred_today = self.current_model.predict(X_today[self.current_top_features])[0]
            y_pred_total.append((current_date, y_today.values[0], y_pred_today))

            # --- Счёт метрик ---
            daily_result = self._evaluate(
                name=model_name,
                y_true=y_today,
                y_pred=np.array([y_pred_today]),
                rate=key_rate_today,
                balance_ref=1.0,  # не важен в онлайн метриках
                extra_metrics=extra_metrics
            )
            daily_result["date"] = current_date
            self.daily_results.append(daily_result)

            metrics_str = ', '.join(f"{k}: {v:.4f}" for k, v in daily_result.items() if k not in ['Model', 'date'])
            print(f"{current_date.date()} | {metrics_str}")

 
            # --- Детекция разладки ---
            # error_today = y_today.values[0] - y_pred_today
            # self.residuals.append(error_today)

            # detected_cps = detector.detect(pd.Series(self.residuals))

            error_today = y_today.values[0] - y_pred_today
            is_cp = detector.update(error_today)

            if is_cp:
                print(f"Разладка на {current_date.date()} — переобучение!")
                self.change_points.append(current_date)

                retrain_data = X[X.index <= current_date]
                retrain_target = y[y.index <= current_date]
                retrain_rate = key_rate[key_rate.index <= current_date]

                self.use_optuna = True
                self._fit(retrain_data, retrain_target, retrain_rate, model_name)
                sunday_counter = 0  # обнуляем счётчик воскресений


            # --- Переобучение по расписанию ---
            if current_date.weekday() == 6:  # Воскресенье
                sunday_counter += 1
                if sunday_counter % reoptimize_every_m_sundays == 0:
                    print(f"Регулярное переобучение с Optuna на {current_date.date()}")
                    retrain_data = X[X.index <= current_date]
                    retrain_target = y[y.index <= current_date]
                    retrain_rate = key_rate[key_rate.index <= current_date]

                    self.use_optuna = True
                    self._fit(retrain_data, retrain_target, retrain_rate, model_name)
                elif sunday_counter % retrain_every_n_sundays == 0:
                    print(f"Регулярное переобучение без Optuna на {current_date.date()}")
                    retrain_data = X[X.index <= current_date]
                    retrain_target = y[y.index <= current_date]
                    retrain_rate = key_rate[key_rate.index <= current_date]

                    self.use_optuna = False
                    self._fit(retrain_data, retrain_target, retrain_rate, model_name)

            # --- Переход к следующему дню ---
            idx = np.where(all_dates == current_date)[0][0]
            if idx + 1 >= len(all_dates):
                break
            current_date = all_dates[idx + 1]

        # --- Финальный расчёт ---
        daily_df = pd.DataFrame(self.daily_results)
        print("\nСредние метрики за период:")
        print(daily_df.select_dtypes(include=[np.number]).mean().T.round(10))

        # --- Построение графика ---
        self._plot_online(y_pred_total)

    def _fit(self, X_train, y_train, rate_train, model_name):
        print(f"Обучение на {len(X_train)} примерах...")

        if self.use_optuna:
            best_params = self.tune_model(model_name, X_train, y_train, rate_train)
        else:
            best_params = {}

        self.current_best_params = best_params
        model_cls = self.models_config[model_name][0]

        # --- обучение selector_model ---
        selector_model = model_cls(**best_params)
        if isinstance(selector_model, CatBoostRegressor):
            from catboost import Pool
            selector_pool = Pool(X_train, y_train)
            selector_model.fit(selector_pool,verbose=0)
        else:
            selector_model.fit(X_train, y_train)

        self.current_top_features = self._select_top_features(selector_model, X_train)

        # --- обучение финальной current_model ---
        self.current_model = model_cls(**best_params)
        if isinstance(self.current_model, CatBoostRegressor):
            from catboost import Pool
            train_pool = Pool(X_train[self.current_top_features], y_train)
            self.current_model.fit(train_pool, verbose=0)
        else:
            self.current_model.fit(X_train[self.current_top_features], y_train)


    def _plot_online(self, y_pred_total):
        dates, y_true, y_pred = zip(*y_pred_total)

        plt.figure(figsize=(15, 6))
        plt.plot(dates, y_true, label="Истинное значение", linewidth=2)
        plt.plot(dates, y_pred, label="Прогноз", linewidth=2)

        for cp in self.change_points:
            plt.axvline(x=cp, color='red', linestyle='--', label='Разладка' if cp == self.change_points[0] else "")

        plt.title("Онлайн прогноз + разладки")
        plt.xlabel("Дата")
        plt.ylabel("Сальдо")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()