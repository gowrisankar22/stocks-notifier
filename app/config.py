from dataclasses import dataclass, field


@dataclass
class MarketConfig:
    id: str
    name: str
    flag: str
    currency: str
    currency_symbol: str
    default_tickers: list[str]


MARKETS: dict[str, MarketConfig] = {
    "ai": MarketConfig(
        id="ai",
        name="AI Stocks (Global)",
        flag="AI",
        currency="Multi",
        currency_symbol="",
        default_tickers=[
            # ── US Large-Cap AI ───────────────────────────────────────
            "NVDA",       # NVIDIA – GPU / AI compute leader
            "GOOGL",      # Alphabet – Gemini, DeepMind, Search AI
            "MSFT",       # Microsoft – Azure AI, Copilot, OpenAI partner
            "META",       # Meta – LLaMA, AI-driven ads & metaverse
            "AMZN",       # Amazon – AWS AI, Alexa, robotics
            "AAPL",       # Apple – On-device ML, Apple Intelligence
            "AVGO",       # Broadcom – AI networking chips
            "AMD",        # AMD – MI300 AI accelerators
            "ORCL",       # Oracle – OCI AI cloud infrastructure
            "TSM",        # TSMC – Fabricates all leading AI chips
            "QCOM",       # Qualcomm – Edge AI / mobile AI chips
            "ADBE",       # Adobe – Firefly generative AI
            "CRM",        # Salesforce – Einstein AI platform
            "IBM",        # IBM – watsonx enterprise AI
            "INTC",       # Intel – Gaudi AI accelerators
            # ── US Mid-Cap AI ─────────────────────────────────────────
            "PLTR",       # Palantir – AIP, government & enterprise AI
            "ARM",        # ARM Holdings – AI-optimized chip architectures
            "CRWD",       # CrowdStrike – AI cybersecurity
            "PANW",       # Palo Alto Networks – AI-driven security
            "SNOW",       # Snowflake – AI data cloud
            "DDOG",       # Datadog – AI observability & monitoring
            "NET",        # Cloudflare – Workers AI, edge inference
            "MRVL",       # Marvell – Custom AI silicon
            "ANET",       # Arista Networks – AI datacenter networking
            "MDB",        # MongoDB – AI app database platform
            "DELL",       # Dell – AI server infrastructure
            "VRT",        # Vertiv – AI datacenter power & cooling
            # ── US Small / Emerging AI ────────────────────────────────
            "CRWV",       # CoreWeave – GPU cloud for AI workloads
            "NBIS",       # Nebius Group – AI cloud (ex-Yandex)
            "APLD",       # Applied Digital – AI datacenter hosting
            "SMCI",       # Super Micro Computer – AI server systems
            "AI",         # C3.ai – Enterprise AI applications
            "PATH",       # UiPath – AI-powered automation
            "SOUN",       # SoundHound AI – Voice AI platform
            "BBAI",       # BigBear.ai – AI analytics for defense
            "UPST",       # Upstart – AI lending platform
            "IONQ",       # IonQ – Quantum computing for AI
            "RGTI",       # Rigetti Computing – Quantum-classical AI
            "ZS",         # Zscaler – AI zero-trust security
            "SYM",        # Symbotic – AI warehouse robotics
            # ── European AI / Semiconductor ───────────────────────────
            "ASML.AS",    # ASML (Netherlands) – EUV lithography monopoly
            "SAP.DE",     # SAP (Germany) – Joule AI copilot, enterprise AI
            "SIE.DE",     # Siemens (Germany) – Industrial AI & digital twin
            "IFX.DE",     # Infineon (Germany) – AI edge semiconductors
            "AIXA.DE",    # AIXTRON (Germany) – Semiconductor deposition
            "DSY.PA",     # Dassault Systèmes (France) – 3D / simulation AI
            "CAP.PA",     # Capgemini (France) – AI consulting & services
            "PRX.AS",     # Prosus (Netherlands) – AI & tech investments
            "BESI.AS",    # BE Semiconductor (Netherlands) – Chip assembly
            "REL.L",      # RELX (UK) – AI-powered data analytics
        ],
    ),
    "us": MarketConfig(
        id="us",
        name="United States",
        flag="US",
        currency="USD",
        currency_symbol="$",
        default_tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"],
    ),
    "de": MarketConfig(
        id="de",
        name="Germany (XETRA)",
        flag="DE",
        currency="EUR",
        currency_symbol="\u20ac",
        default_tickers=["SAP.DE", "SIE.DE", "ALV.DE", "BAS.DE", "DTE.DE", "BMW.DE", "ADS.DE"],
    ),
    "fr": MarketConfig(
        id="fr",
        name="France (Euronext Paris)",
        flag="FR",
        currency="EUR",
        currency_symbol="\u20ac",
        default_tickers=["MC.PA", "OR.PA", "SAN.PA", "AI.PA", "BNP.PA", "SU.PA", "AIR.PA"],
    ),
    "nl": MarketConfig(
        id="nl",
        name="Netherlands (Euronext)",
        flag="NL",
        currency="EUR",
        currency_symbol="\u20ac",
        default_tickers=["ASML.AS", "PHIA.AS", "UNA.AS", "INGA.AS", "AD.AS", "HEIA.AS"],
    ),
    "uk": MarketConfig(
        id="uk",
        name="United Kingdom (LSE)",
        flag="GB",
        currency="GBP",
        currency_symbol="\u00a3",
        default_tickers=["SHEL.L", "AZN.L", "HSBA.L", "ULVR.L", "GSK.L", "BP.L", "RIO.L"],
    ),
    "it": MarketConfig(
        id="it",
        name="Italy (Borsa Italiana)",
        flag="IT",
        currency="EUR",
        currency_symbol="\u20ac",
        default_tickers=["ENI.MI", "ISP.MI", "UCG.MI", "ENEL.MI", "STLAM.MI", "RACE.MI"],
    ),
    "es": MarketConfig(
        id="es",
        name="Spain (BME)",
        flag="ES",
        currency="EUR",
        currency_symbol="\u20ac",
        default_tickers=["SAN.MC", "ITX.MC", "IBE.MC", "BBVA.MC", "TEF.MC", "REP.MC"],
    ),
    "ch": MarketConfig(
        id="ch",
        name="Switzerland (SIX)",
        flag="CH",
        currency="CHF",
        currency_symbol="CHF\u00a0",
        default_tickers=["NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ABBN.SW", "ZURN.SW"],
    ),
}

DEFAULT_MARKET = "ai"


@dataclass
class Settings:
    app_name: str = "Stock Analyzer Pro"
    refresh_interval_seconds: int = 30
    cache_ttl_seconds: int = 60
    default_watchlist: list[str] = field(
        default_factory=lambda: MARKETS[DEFAULT_MARKET].default_tickers.copy()
    )
    history_period_short: str = "6mo"
    history_period_long: str = "2y"
    ml_prediction_enabled: bool = True


settings = Settings()
