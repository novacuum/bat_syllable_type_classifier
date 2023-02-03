

def latex_statistical_number(value, sd, use_percent):
    return f'\\SI{{{value*100:.1f} \\pm {sd*100:.1f}}}{{\\percent}}' if use_percent else f'\\SI{{{value:.0f} \\pm {sd:.1f}}}{{}}'
