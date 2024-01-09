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
    #       - Sollte die Minütlichen Vorhersagen n - m mal hintereinander
    #         in Folge steigen / fallen, dann ist das ein Indiz dafür,
    #         dass eine News den Trend beeinflusst. Hierbei steht n für die
    #         maximale Anzahl an aufeinander folgender steigender / fallender
    #         Vorhersagen und m für eine dynamische Variable welche das
    #         Verhältnis zwischen n und m bestimmt. Je höher n, desto
    #         höher m. Je niedriger n, desto niedriger m. m kann auch
    #         0 sein, wodurch n beliebig hoch sein kann.
    #   - VALIDATION
    #       - Trend Validierung mit einem LLM
    #       - Trend Analyse

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
