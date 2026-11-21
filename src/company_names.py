"""Name / product / executive terms per ticker for the relevance filter.

Covers the 89 tickers in the FNSPID Nasdaq-100 (2019+) slice. Only distinctive terms are
listed: brand names that double as common words ("Zoom", "Vertex", "Prime", "Windows")
are given in an unambiguous form ("Zoom Video", "Prime Video") or left out. The ticker
symbol is always added automatically; symbols that are also English words are listed in
AMBIGUOUS_SYMBOLS and only match as "$ON", "(ON)" or "NASDAQ:ON".
"""

from typing import Dict, List, Set

COMPANY_TERMS: Dict[str, List[str]] = {
    "AAPL": ["Apple", "iPhone", "iPad", "MacBook", "Tim Cook", "App Store", "Vision Pro", "AirPods", "Apple Watch"],
    "ADBE": ["Adobe", "Photoshop", "Figma"],
    "ADI": ["Analog Devices"],
    "ADP": ["Automatic Data Processing"],
    "ADSK": ["Autodesk"],
    "AEP": ["American Electric Power"],
    "ALGN": ["Align Technology", "Invisalign"],
    "AMAT": ["Applied Materials"],
    "AMD": ["Advanced Micro Devices", "Ryzen", "Radeon", "Lisa Su", "Xilinx"],
    "AMGN": ["Amgen"],
    "AMZN": ["Amazon", "AWS", "Bezos", "Andy Jassy", "Prime Video", "Alexa", "Whole Foods"],
    "ANSS": ["Ansys"],
    "ASML": ["ASML"],
    "AVGO": ["Broadcom"],
    "AZN": ["AstraZeneca"],
    "BIIB": ["Biogen", "Aduhelm", "Leqembi"],
    "BKNG": ["Booking Holdings", "Booking.com", "Priceline"],
    "BKR": ["Baker Hughes"],
    "CDNS": ["Cadence Design"],
    "CHTR": ["Charter Communications"],
    "CMCSA": ["Comcast", "NBCUniversal", "Xfinity"],
    "COST": ["Costco"],
    "CPRT": ["Copart"],
    "CRWD": ["CrowdStrike"],
    "CSCO": ["Cisco"],
    "CSGP": ["CoStar"],
    "CSX": ["CSX"],
    "CTAS": ["Cintas"],
    "CTSH": ["Cognizant"],
    "DDOG": ["Datadog"],
    "DLTR": ["Dollar Tree", "Family Dollar"],
    "DXCM": ["Dexcom", "DexCom"],
    "EA": ["Electronic Arts"],
    "EBAY": ["eBay"],
    "ENPH": ["Enphase"],
    "EXC": ["Exelon"],
    "FANG": ["Diamondback Energy"],
    "FAST": ["Fastenal"],
    "FTNT": ["Fortinet"],
    "GILD": ["Gilead"],
    "GOOG": ["Alphabet", "Google", "YouTube", "Waymo", "Sundar Pichai", "DeepMind"],
    "GOOGL": ["Alphabet", "Google", "YouTube", "Waymo", "Sundar Pichai", "DeepMind"],
    "IDXX": ["Idexx", "IDEXX"],
    "ILMN": ["Illumina"],
    "INTC": ["Intel", "Pat Gelsinger"],
    "INTU": ["Intuit", "TurboTax", "QuickBooks", "Credit Karma", "Mailchimp"],
    "JD": ["JD.com"],
    "KHC": ["Kraft Heinz", "Kraft", "Heinz"],
    "KLAC": ["KLA Corp", "KLA-Tencor", "KLA"],
    "LRCX": ["Lam Research"],
    "LULU": ["Lululemon"],
    "MAR": ["Marriott"],
    "MCHP": ["Microchip Technology"],
    "MDLZ": ["Mondelez", "Oreo", "Cadbury"],
    "MELI": ["MercadoLibre", "Mercado Libre"],
    "MNST": ["Monster Beverage"],
    "MRVL": ["Marvell"],
    "MSFT": ["Microsoft", "Azure", "Satya Nadella", "Xbox", "LinkedIn"],
    "MU": ["Micron"],
    "NFLX": ["Netflix"],
    "NVDA": ["Nvidia", "NVIDIA", "Jensen Huang", "GeForce"],
    "NXPI": ["NXP"],
    "ODFL": ["Old Dominion"],
    "ON": ["ON Semiconductor", "onsemi"],
    "ORLY": ["O'Reilly Automotive", "O’Reilly Automotive"],
    "PANW": ["Palo Alto Networks"],
    "PAYX": ["Paychex"],
    "PCAR": ["Paccar", "PACCAR", "Kenworth", "Peterbilt"],
    "PDD": ["Pinduoduo", "Temu"],
    "PEP": ["PepsiCo", "Pepsi", "Frito-Lay"],
    "PYPL": ["PayPal", "Venmo"],
    "QCOM": ["Qualcomm", "Snapdragon"],
    "REGN": ["Regeneron", "Dupixent"],
    "ROST": ["Ross Stores"],
    "SBUX": ["Starbucks"],
    "SGEN": ["Seagen", "Seattle Genetics"],
    "SIRI": ["Sirius XM", "SiriusXM"],
    "SNPS": ["Synopsys"],
    "TEAM": ["Atlassian"],
    "TMUS": ["T-Mobile"],
    "TSLA": ["Tesla", "Cybertruck", "Model 3", "Model Y", "Model S", "Model X",
             "Gigafactory", "Giga Shanghai", "Giga Berlin", "Giga Texas", "Autopilot", "Full Self-Driving", "FSD",
             "Supercharger", "Powerwall", "Megapack", "Solar Roof", "SolarCity"],
    "TXN": ["Texas Instruments"],
    "VRSK": ["Verisk"],
    "VRTX": ["Vertex Pharmaceuticals"],
    "WBA": ["Walgreens"],
    "WDAY": ["Workday"],
    "XEL": ["Xcel Energy"],
    "ZM": ["Zoom Video"],
    "ZS": ["Zscaler"],
}

# Ticker symbols that are also ordinary words / common abbreviations.
AMBIGUOUS_SYMBOLS: Set[str] = {"ON", "FAST", "TEAM", "MAR", "FANG", "COST"}

# Terms that only count when the headline is not about the person's *other* ventures
# (e.g. "Musk" headlines about Twitter/X or SpaceX are not Tesla news).
WEAK_TERMS: Dict[str, List[str]] = {
    "TSLA": ["Elon Musk", "Musk"],
}
WEAK_TERM_EXCLUSIONS: Dict[str, List[str]] = {
    "TSLA": ["Twitter", "SpaceX", "X logo", "X Corp", "xAI", "Neuralink", "Starlink", "Boring Company",
             "Synchron", "Grimes", "space station", "space tourism"],
}
