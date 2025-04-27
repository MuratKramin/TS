import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from ModelPipeline import ModelPipeline

class OnlineForecastRunner:
    def __init__(self, model_pipeline, cusum_detector, retrain_every_n_weeks=4, optuna_every_m_weeks=12,extra_metrics=None):
        self.model_pipeline = model_pipeline
        self.detector = cusum_detector
        self.retrain_every_n_weeks = retrain_every_n_weeks
        self.optuna_every_m_weeks = optuna_every_m_weeks
        self.retrain_counter = 0
        self.optuna_retrain_counter = 0
        self.history = []
        self.change_points = []
        
        self.extra_metrics = extra_metrics or []


    def run(self, X, y, key_rate, start_date, model_name):
        dates = X.loc[X.index >= start_date].index
        current_X = X.loc[X.index < start_date].copy()
        current_y = y.loc[y.index < start_date].copy()
        current_rate = key_rate.loc[key_rate.index < start_date].copy()

        # Первичное обучение с Optuna
        self.model_pipeline.use_optuna = True
        self.model_pipeline.best_params = {}  # Сброс параметров
        self._train_model(current_X, current_y, current_rate, model_name)

        predictions = []
        reals = []

        for current_date in dates:
            X_today = X.loc[[current_date]]
            y_today = y.loc[[current_date]]
            rate_today = key_rate.loc[[current_date]]

            y_pred = self.model.predict(X_today[self.top_features])[0]
            metric_results = self._calculate_metrics(y_today, y_pred, rate_today)

            metrics_str = ", ".join(f"{k}: {v:.5f}" for k, v in metric_results.items())
            print(f"{current_date.date()} — {metrics_str}")

            # запись прогноза
            self.history.append({
                "date": current_date,
                "real": y_today.values[0],
                "pred": y_pred,
                **metric_results
            })

            predictions.append(y_pred)
            reals.append(y_today.values[0])

            # --- детекция разладки ---
            detected_cp = self.detector.detect(pd.Series([y_today.values[0] - y_pred]))
            if detected_cp:
                print(f"ыРазладка обнаружена на {current_date.date()}! Переобучение с Optuna.")
                self.model_pipeline.use_optuna = True
                self.model_pipeline.best_params = {}
                self._train_model(X.loc[X.index <= current_date], y.loc[y.index <= current_date], key_rate.loc[key_rate.index <= current_date], model_name)
                self.change_points.append(current_date)

            # --- еженедельные переобучения ---
            if current_date.weekday() == 6:  # Воскресенье
                self.retrain_counter += 1
                self.optuna_retrain_counter += 1

                if self.optuna_retrain_counter >= self.optuna_every_m_weeks:
                    print(f"Плановое переобучение с Optuna на {current_date.date()}.")
                    self.model_pipeline.use_optuna = True
                    self.model_pipeline.best_params = {}
                    self._train_model(X.loc[X.index <= current_date], y.loc[y.index <= current_date], key_rate.loc[key_rate.index <= current_date], model_name)
                    self.optuna_retrain_counter = 0
                    self.change_points.append(current_date)
                elif self.retrain_counter >= self.retrain_every_n_weeks:
                    print(f"Плановое переобучение без Optuna на {current_date.date()}.")
                    self.model_pipeline.use_optuna = False
                    self._train_model(X.loc[X.index <= current_date], y.loc[y.index <= current_date], key_rate.loc[key_rate.index <= current_date], model_name)
                    self.retrain_counter = 0
                    self.change_points.append(current_date)

        self._plot_results()
        return pd.DataFrame(self.history)

    def _train_model(self, X_train, y_train, rate_train, model_name):
        # Без разделения на тест, всё на обучении
        run_result = self.model_pipeline.run(
            X_train,
            y_train,
            rate_train,
            test_start_date=X_train.index[-1] + pd.Timedelta(days=1),
            model_name=model_name,
            extra_metrics=[]
        )
        # сохраняем модель
        self.model = self.model_pipeline.models_config[model_name][0](**self.model_pipeline.best_params[model_name])

        # здесь правильный способ получить топовые признаки:
        selector_model = self.model_pipeline.models_config[model_name][0](**self.model_pipeline.best_params[model_name])
        selector_model.fit(X_train, y_train)
        self.top_features = self.model_pipeline._select_top_features(selector_model, X_train)

        self.model.fit(X_train[self.top_features], y_train)


    def _calculate_metrics(self, y_true, y_pred, rate):
        res = {
            "MAE": mean_absolute_error(y_true, [y_pred]),
            self.model_pipeline.metric_func.__name__: self.model_pipeline.metric_func(y_true, [y_pred], rate),
        }
        #for func in self.model_pipeline.extra_metrics:
        for func in self.extra_metrics:

            res[func.__name__] = func(y_true, y_pred, rate)
        return res

    def _plot_results(self):
        df_history = pd.DataFrame(self.history)
        plt.figure(figsize=(15, 6))
        plt.plot(df_history["date"], df_history["real"], label="Факт", linewidth=2)
        plt.plot(df_history["date"], df_history["pred"], label="Прогноз", linewidth=2)
        for cp in self.change_points:
            plt.axvline(cp, color="red", linestyle="--", alpha=0.7)
        plt.xlabel("Дата")
        plt.ylabel("Сальдо")
        plt.title("Онлайн-прогноз с разладками и переобучениями")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

df = pd.read_csv("data/merged_dataset_with_lags_rollings.csv", parse_dates=["date"]).set_index("date")
X = df.drop(columns=["balance", "income", "outcome"])
y = df["balance"]
key_rate = df["rate"]#.fillna(0.085)

pipeline = ModelPipeline(
    models_config,
    metric_func=mae_cust,  # или awmae
    optuna_direction="minimize",
    n_splits=2,
    n_trials=1,
    #extra_metrics=[calculate_add_margin_vectorized, delta_pnl]
)

runner = OnlineForecastRunner(
    model_pipeline=pipeline,
    cusum_detector=UniversalCUSUMDetector(mode="adaptive", threshold=5.0, drift=0.01),
    retrain_every_n_weeks=4,
    optuna_every_m_weeks=12,
    extra_metrics=[calculate_add_margin_vectorized, delta_pnl]
)

online_results = runner.run(
    X=X,
    y=y,
    key_rate=key_rate,
    start_date=pd.Timestamp("2020-09-01"),  # твоя стартовая дата
    model_name="LGBM"
)

print(online_results.mean())
