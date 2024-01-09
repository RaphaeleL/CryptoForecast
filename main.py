#!/usr/bin/env python3

from forecast import CryptoForecast


def main(cf):

    # TODO
    #   - NEWS
    #       - Ein weiteres Feature signalisiert den Trend, wodurch
    #         News (also der Trend) beeinflusst werden kann.
    #       - Minütliche Daten der letzten 7 Tage sind mit Yahoo möglich.
    #         Damit kann eine News zwar nicht eingearbeitet werden. Jedoch,
    #         kann rechtzeitig die Folgen erkannt werden um entsprechend,
    #         mit der Vorhersage darauf einzugehen. Eine Vorhersage MUSS
    #         weniger als 60 Sekunden dauern.
    #   - VALIDATION
    #       - Trend Analysis (Up, Down, Sideways, etc.)
    #       - LLMs (ChatGPT, LLaMa, etc.))
    #       - For Step 3

    if cf.should_retrain:
        cf.load_history()
        cf.stop_time("to retrain the model")
        exit()

    if cf.args.agents:
        results = cf.use_agents()
        cf.stop_time("to predict the future with Agents.")
        cf.visualize_agents(results)
    else:
        cf.load_history()
        cf.predict_future()
        cf.stop_time("to predict the future with Weights.")
        cf.visualize()


if __name__ == "__main__":
    main(CryptoForecast())
