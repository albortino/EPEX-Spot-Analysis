# ⚡ Strompreis-Analyse Dashboard

Ein interaktives Streamlit-Dashboard zur datengestützten Analyse des eigenen Stromverbrauchs und zum präzisen Tarifvergleich.

> **Kernziel des Projekts:**  
> Analysiere anhand deines **tatsächlichen, historischen Verbrauchsverhaltens**, ob **dynamische/flexible Stromtarife (EPEX Spot)** für dich günstiger sind als **klassische statische Tarife (fester Arbeitspreis pro kWh)** – ganz ohne Schätzwerte oder Rätselraten.

---

## 🚀 Live-Demo & Eigenbetrieb

- **Online nutzen:** Das Dashboard wird öffentlich gehostet – du kannst direkt im Browser deine Verbrauchsdaten hochladen und sofort loslegen. Die Daten werden vom Server verarbeitet aber nicht gespeichert! *(Link folgt / hier einfügen)*
- **Self-Hosting / Lokal ausführen:** Du möchtest vollständige Kontrolle haben oder es weiterentwickeln? Die Anwendung lässt sich mit wenigen Befehlen vollständig lokal auf deinem eigenen Rechner ausführen.

---

## Funktionen des Dashboards

Viele Haushalte überlegen, zu einem dynamischen Börsenstromtarif (z. B. Tibber, aWATTar, etc.) zu wechseln. Doch lohnt sich das bei deinem individuellen Lastprofil und den aktuellen Strompreisen wirklich?

Dieses Dashboard verschneidet deine historischen Smart-Meter-Messwerte (typischerweise als CSV) mit den echten stündlichen **EPEX-Spot-Börsenpreisen** deines Landes und liefert dir glasklare Antworten:

* **Echter Kostenvergleich:** Berechnet auf den Cent genau, was dich dein Strom im vergangenen Jahr mit flexiblen vs. statischen Tarifen gekostet hätte.
* **Automatische Tarifempfehlung:** Erkennt automatisch den günstigsten vordefinierten statischen und flexiblen Tarif für dein individuelles Profil.
* **Eigene Tarife simulieren:** Hinterlege eigene Arbeitspreise, Grundgebühren, Aufschläge oder prozentuale Börsenaufschläge deines Wunschanbieters.
* **Lastverschiebung simulieren (Peak Load Shifting):** Was bringt es, wenn du z. B. 20 % deines Spitzenverbrauchs (z. B. E-Auto laden, Waschmaschine) in günstige Stunden verschiebst? Der Schieberegler zeigt dir das Einsparpotenzial in Euro.
* **Detaillierte Lastprofil-Analyse:** Automatische Aufteilung deines Verbrauchs in Grundlast (`Base Load`), Regellast (`Regular Load`) und Spitzenlast (`Peak Load`).
* **Interaktive Visualisierungen:**
  * **Spotpreis-Heatmap:** Wann ist Strom typischerweise günstig oder teuer? (Uhrzeiten & Saisonalität).
  * **Verbrauchs-Trends & Beispieltage:** Analysiere einzelne Tage oder langfristige Muster.
  * **Jahresübersicht:** Detaillierter Jahresvergleich von Kosten und Verbrauch.
* **Datenexport:** Exportiere bereinigte Analysedaten und EPEX-Spotpreise als Excel-Datei (`.xlsx`) für eigene Auswertungen.
* **Urlaubstage filtern:** Schließe Abwesenheitstage automatisch aus, um verzerrende Tage ohne Normalverbrauch herauszurechnen.

---

## Hinweise

* **Privacy First:** Deine hochgeladenen CSV-Dateien werden ausschließlich im flüchtigen Speicher der aktuellen Browser-Sitzung verarbeitet. Es werden keine Benutzerkonten angelegt und keine Verbrauchshistorien gespeichert.
* **Berechnungsumfang sind reine Energiekosten:** Das Dashboard berechnet die reinen **Energiekosten** inklusive Anbieter-Aufschlägen und monatlichen Grundgebühren. Netzentgelte, Steuern, Abgaben und staatliche Umlagen variieren je nach Netzbetreiber/Region und sind in den Musterberechnungen standardmäßig nicht enthalten.
* **Keine Anlage- oder Wechselberatung:** Die hinterlegten Tarife dienen als Orientierungs- und Vergleichskatalog. Bitte prüfe die tagesaktuellen Konditionen direkt beim jeweiligen Anbieter (oder dem Tarifkalkulator des E-Control (https://www.e-control.at/tarifkalkulator#/)) vor Vertragsschluss. Es gibt kein Sponsoring, weshalb volle Transparenz gewährleistet wird.

---

## Datenanforderungen

* **Verbrauchs-CSV:** Eine CSV-Datei mit deinem historischen Stromverbrauch (z. B. aus dem Kundenportal deines Netzbetreibers oder Smart-Meter-Gateways). Ideal sind 15-Minuten- oder 1-Stunden-Werte. Das Tool erkennt gängige Netzbetreiber- und Anbieterformate automatisch.
* **Börsenstrompreise:** Werden für das gewählte Land (z. B. Deutschland, Österreich) und den ausgewählten Zeitraum vollautomatisch über die aWATTar-API abgerufen und lokal gecacht.

---

## Lokale Installation

### 1. Repository klonen
```bash
git clone <repository_url>
cd strompreis-analyse
```

### 2. Umgebung einrichten
Mit Conda (empfohlen):
```bash
conda env create -f environment.yml
conda activate strompreis-analyse
```
*Alternativ mit `pip`: `pip install -r requirements.txt` (sofern vorhanden).*

### 3. Dashboard starten
```bash
streamlit run app.py
```
Die Anwendung öffnet sich automatisch unter `http://localhost:8501`.



## 📁 Projektstruktur

```text
├── app.py                  # Streamlit-Haupteinstiegspunkt & UI-Orchestrierung
├── methods/
│   ├── config.py           # Globale Konfigurationen & Konstanten
│   ├── data_loader.py      # EPEX-Spotpreis-Abruf (aWATTar API) & Caching
│   ├── file_parser.py      # Automatischer Parser für verschiedene Smart-Meter-CSV-Formate
│   ├── analysis.py         # Analyselogik (Lastprofil-Klassifizierung, Lastverschiebung)
│   ├── tariffs.py          # Tarifverwaltung und Tarif-Kostenberechnungslogik
│   ├── ui_components.py    # Streamlit-UI-Komponenten, Tabs & Seitenleiste
│   ├── charts.py           # Plotly-Visualisierungen und Charts
│   └── utils.py            # Hilfsfunktionen (Datumskonvertierung, Excel-Export)
└── cache/                  # Lokaler Cache für abgerufene Preisdaten
```

---

## Danksagung

Dieses Projekt basiert auf Ideen und Parser-Logiken des [awattar backtesting](https://awattar-backtesting.github.io/)-Projekts und erweitert dieses um moderne Auswertungen, Peak-Shifting-Simulationen und ein interaktives Dashboard.
