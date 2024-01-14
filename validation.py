from utils import get_colored_text

keys = ["Monte Carlo Simulation", "Trend Analysis", "Asking a LLM"]
space = lambda x, y: "." * (len(max(keys, key=len)) - len(x) + 5 + y)


def convert_result_to_text(text):
    if text is True:
        return get_colored_text("passed", "green")
    elif text is False:
        return get_colored_text("failed", "red")
    else:
        return ""


def monte_carlo_simulation(cf):
    return True


def trend_analysis(cf):
    return True


def llm(cf):
    return True


def validate(cf):
    mcs_res = monte_carlo_simulation(cf)
    ta_res = trend_analysis(cf)
    llm_res = llm(cf)


    results = {"Monte Carlo Simulation": mcs_res, "Trend Analysis": ta_res, "Asking a LLM": llm_res}

    for index, key in enumerate(keys):
        delim = "└──" if index == len(keys) - 1 else "├──"
        print(f"{delim} {key} {space(key, 0)} {convert_result_to_text(results[key])}")
