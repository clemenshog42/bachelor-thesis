# 📘 Regelwerk für die Bachelorarbeit (KI & Autor Leitfaden)

Dieses Dokument dient als striktes Regelwerk und Master-Prompt für das Verfassen der Bachelorarbeit. Es ist sowohl für den menschlichen Autor als auch für KI-Assistenten konzipiert, die bei der Erstellung unterstützen. Alle Vorgaben aus den Richtlinien des Betreuers (G. Zwettler) sowie der FH OÖ sind hier integriert.

---

## 🎯 1. Spezifika und Fokus der Arbeit

- **Titel:** Zeitlich robuste Erkennung und Zustandsmodellierung von Katzen und Beutetieren mittels Deep Learning
- **Problemstellung:** Objektdetektoren (wie YOLO) sind auf Einzelbildern gut, aber anfällig bei Verdeckungen/Bewegungsunschärfe. Ziel ist es, zuverlässig zu erkennen, ob eine Katze ein Beutetier (Maus/Vogel) trägt, was eine Analyse über mehrere Frames erfordert.
- **Methodik-Fokus:** 
  1. Objektdetektor (YOLOv11) für Echtzeit-Inferenz.
  2. Tracking-by-Detection (ByteTrack) zur Verfolgung über Frames und Vermeidung von Identitätswechseln.
  3. Bildklassifikation (YOLOv11-Klassifikation).
  4. Zustandsautomat (State Machine), der zeitlich aggregierte Konfidenzwerte nutzt, um Zustände wie "Katze mit Beute" robust zu modellieren und Fehlalarme zu reduzieren.
- **Ziel/Ergebnis:** Ein robusteres System im Vergleich zur Einzelbild-Klassifikation.

---

## 📊 2. Bewertungskriterien (Ziele für jedes Kapitel)

Die Arbeit wird strikt nach folgendem FH OÖ Beurteilungsschema bewertet. Diese Aspekte müssen in der Arbeit jederzeit maximiert werden:

### 🧠 Inhaltliche Aspekte (40 % der Note)
- Klare und präzise Darstellung der Problematik, Fragestellung und Arbeitshypothese.
- Das Thema wird adäquat und in der nötigen Tiefe behandelt.
- Zusammenfassung und Ausblick sind logisch und spiegeln die Ergebnisse wider.
- Hohe inhaltliche Eigenständigkeit (klarer Unterschied zwischen bestehendem Wissen und eigener Leistung).

### 🔬 Methodische Aspekte (30 % der Note)
- Nachvollziehbare Darlegung und fundierte Begründung der gewählten methodischen Vorgangsweise.
- Systematische Art und Weise der Aufarbeitung des Wissens.
- Saubere Auswertung und neutrale Darstellung der Ergebnisse.
- Hohe Qualität, Streuung und Aktualität der Literatur (Primärliteratur!).
- Roter Faden: Logische und stringente Abfolge der Kapitel.
- Methodische Eigenständigkeit.

### 📐 Formale Aspekte (30 % der Note)
- Einwandfreie Gliederung (siehe Kapitelstruktur).
- Hochwertige sprachliche Aspekte (technisch, sachlich, gendergerecht).
- Professionelles Layout.
- Korrekte Zitation und sauberes Literaturverzeichnis ohne Ausnahmen.

---

## 🏗 3. Struktur und Aufbau der Arbeit

Der rote Faden und die Kapitelstruktur folgen strikten Vorgaben zu Inhalt und Umfang:

### 1. Einleitung – ca. 4–5 Seiten
Die Einleitung führt in das Thema ein und erläutert, warum die bearbeitete Problemstellung relevant ist.
- **Inhalte:** Motivation und praktische/wissenschaftliche Relevanz, Problemstellung, Zielsetzung der Arbeit, Forschungsfragen, kurzer Überblick über den Stand der Technik (Related Work), Abgrenzung der Arbeit, eigener Lösungsansatz in groben Zügen, Überblick über den Aufbau der Arbeit.
- *Hinweis:* Noch nicht detailliert erklären, wie die Algorithmen funktionieren. Dient nur als Hinführung.

### 2. Theoretische Grundlagen – ca. 5–7 Seiten
Dieses Kapitel enthält ausschließlich die theoretischen Grundlagen, die für das Verständnis der *eigenen* Arbeit zwingend notwendig sind.
- **2.1 Objekterkennung mit YOLOv11:** Erklärung nur der Konzepte, die für das Verständnis benötigt werden (Was ist Objekterkennung? Unterschied zur Klassifikation, Grundprinzip eines One-Stage-Detektors, wie YOLO grundsätzlich arbeitet, was YOLOv11 im Kontext macht, Ausgabe wie Bounding Boxes und Klassen, ggf. Confidence Score und IoU). *Keine detaillierte mathematische Herleitung des kompletten Netzwerks!*
- **2.2 Multi-Object Tracking mit ByteTrack:** Was ist das? Warum reicht Erkennung nicht? Grundidee Tracking-by-Detection, Detektionen über mehrere Frames zuordnen, Grundprinzip von ByteTrack und warum es hier geeignet ist. *Keine vollständige mathematische Herleitung.*
- **2.3 Bildklassifikation mit YOLOv11:** Was ist Bildklassifikation? Unterschied zur Detektion, Grundprinzip von CNNs, Nutzung von YOLO für Klassifikationsaufgaben und warum wurde es verwendet? Ein- und Ausgabe des Modells. *Keine detaillierte Erklärung jeder Faltungsschicht, da nur vortrainiert genutzt.*
- **2.4 Zustandsautomaten (Finite State Machines):** Ausführlicher, da zentraler Bestandteil. Was ist ein Zustandsautomat? (Zustände, Übergänge, Ereignisse, Aktionen, Übergangslogik). Ggf. ein einfaches allgemeines Beispiel.
- *Wichtig:* Verworfene Algorithmen gehören nicht hierher! Mathematische Grundlagen nur so tief wie zwingend nötig. Die konkrete FSM-Umsetzung kommt erst in Kapitel 4.

### 3. Material / Versuchsgrundlage – ca. 3–5 Seiten
Beschreibt die konkreten Ressourcen und Ausgangsdaten.
- **Inhalte:** 
  - Verwendete Testdaten und Datensätze (Katzen mit Beutetieren wurden KI-generiert, für die Gesichtserkennung wurde das Kaggle-Dataset `https://www.kaggle.com/datasets/crawford/cat-dataset` verwendet).
  - Herkunft, Eigenschaften, Anzahl und Art der Daten, Versuchsgruppen.
  - Verwendete Hardware (Kameras, Rechner/GPU).
  - Softwareumgebung (Frameworks, Bibliotheken, vortrainierte Modelle).

### 4. Methodik und Entwicklung – ca. 10–14 Seiten
Zentrales Kapitel (neben Kapitel 6 größter Umfang). Detaillierte Beschreibung der Entwicklung.
- **Inhalte:** Gesamtarchitektur, Datenfluss, Erkennung mit YOLO (Anpassung/Konfiguration), Tracking mit ByteTrack (Parametrisierung), Klassifikation mit YOLO, Entwicklung und Logik des Zustandsmodells (Definition der Zustände und Übergänge, Zusammenwirken der Komponenten, Behandlung von Sonderfällen). Entscheidungen während der Entwicklung, iterative Verbesserungen, Begründungen. (Keras/EfficientNet kann als verworfener Ansatz kurz erwähnt werden mit Begründung).

### 5. Design und Implementierung – ca. 4–6 Seiten
Beschreibt die konkrete technische Umsetzung (Wie wurde das Konzept aus Kap. 4 technisch realisiert?).
- **Inhalte:** Softwarearchitektur, Struktur der Komponenten, Schnittstellen, zentrale Klassen und Module, Datenstrukturen, Implementierung des Zustandsmodells, relevante Konfigurationen.
- *Wichtig:* Keine reine Code-Dokumentation! Klare Trennung: Kap. 4 = Konzept, Kap. 5 = Technische Umsetzung.

### 6. Tests und Analysen – ca. 8–12 Seiten
Zweites zentrales Kapitel. Nüchterne Präsentation der Ergebnisse.
- **Inhalte:** Testaufbau, Testfälle, Evaluationskriterien/Metriken, quantitative und qualitative Ergebnisse, Vergleich von Konfigurationen. Analyse der Erkennungsleistung, Tracking-Leistung und des Zustandsmodells. Fehlerfälle, Robustheit, Laufzeit.
- *Hinweis:* Ergebnisse nur nüchtern präsentieren ("Konfiguration A erreichte X %"), noch nicht interpretieren.

### 7. Diskussion – ca. 4–7 Seiten
Interpretation und kritische Bewertung der Ergebnisse aus Kapitel 6.
- **Inhalte:** Beantwortung der Forschungsfragen, Bewertung des Ansatzes (Erfolgreich? Ziele erreicht/nicht erreicht?), Ursachen für Ergebnisse (Warum hat Konfiguration A besser funktioniert?), Stärken, Schwächen, Limitationen, mögliche Fehlerquellen, Vergleich mit dem Stand der Technik.

### 8. Zusammenfassung und Ausblick – ca. 2–3 Seiten
- **Inhalte:** Zentrale Erkenntnisse, kurze Beantwortung der Forschungsfragen, wichtigste Ergebnisse, zentrale Beiträge, abschließende Bewertung, mögliche Weiterentwicklungen (Optimierungsmöglichkeiten). Nicht die ganze Arbeit wiederholen!

### 🚫 Anti-Patterns bei der Gliederung (VERBOTEN):
- **Keine kreativen Überschriften:** Keine Titel wie "Die Hardware schlägt zurück". Sachlich bleiben.
- **Keine Redundanz:** Keine Überschriften wie `6.1 Algorithmus A` gefolgt von `6.1.1 Ergebnisse Algorithmus A`.
- **Ausgewogenheit:** Kein Unterkapitel `2.1.1`, wenn es kein `2.1.2` gibt. Maximal drei Ebenen tief (z.B. `1.1.1`).
- **Sonderzeichen-Verbot:** Keine Formeln, Referenzen oder Sonderzeichen in den Überschriften.
- **Einleitende Worte:** Nach einer Hauptüberschrift (z.B. `2. Grundlagen`) muss ein einleitender Text folgen, bevor `2.1 Unterkapitel` beginnt.
- **Logische Gruppierung:** Kapitel auf derselben Ebene müssen denselben Detailgrad und eine logische Klammer aufweisen.

---

## ✍️ 4. Schreibstil und Tonalität

Die Bachelorarbeit ist ein wissenschaftliches, technisches Dokument und kein Roman.
- **Präsens nutzen:** Eigene Entwicklungen und zeitlose Fakten im Präsens schreiben. Vergangenheit nur bei historischen Rückblicken.
- **Niemals ICH-Form:** Die Arbeit wird völlig sachlich, steril und neutral verfasst.
- **Keine Erlebnisberichte / Floskeln:** Keine Formulierungen wie "Da Methode A schlecht war, habe ich B probiert". Stattdessen: "Da Methode A aus Grund X nicht die erforderlichen Ergebnisse erzielt, wird auf Methode B zurückgegriffen."
- **Begriffe konsistent halten:** Nicht variieren! Wenn der Begriff "Objektdetektor" eingeführt wurde, nicht plötzlich "Suchverfahren" schreiben.
- **Quantifizieren statt qualifizieren:** Nicht "Python ist leichter als C++", sondern konkrete Metriken nutzen (z.B. "reduziert die Laufzeit um 12.3% [±4.1]").
- **Formatierung:** 
  - Keine exzessiven Klammern im Text.
  - Textauszeichnungen: **Niemals Fett (bold)** für Hervorhebungen nutzen! Nur *kursive* Schreibweise für Hervorhebungen oder `texttt` für Code-Variablen (z.B. `x` oder `getShortestPath`).

---

## 📚 5. Zitierregeln und Quellenarbeit

Das korrekte Zitieren ist essentiell, Plagiatsprüfungen sind streng (< 5% Übereinstimmung gefordert).
- **KEINE Absatzzitate:** Direkte Zitate (Copy & Paste Text) sind in der Informatik unüblich. Inhalte in eigenen Worten wiedergeben.
- **Zitierposition:** Die Referenz steht *immer* innerhalb des Satzes vor dem Punkt (z.B. "...stellt eine Verbesserung dar \cite{...}."). 
- **Keine Referenzen = Eigene Leistung:** Jeder Satz ohne Referenz wird als eigene intellektuelle Leistung gewertet. Unbelegte Fakten sind unzulässig!
- **Zitierstil:** Vancouver. Muss durchgehend konsistent sein.
- **Keine Fußnoten für Literatur:** Fußnoten nur für URLs nutzen.
- **Hochwertige Literatur:** Ausschließlich Primärliteratur nutzen. Keine Wikipedia. 

---

## 🤖 6. Umgang mit KI-Systemen

- **Deklaration:** Die Nutzung von KI muss im Dokument ehrlich deklariert werden.
- **KI als Autor verboten:** Die zentralen intellektuellen Leistungen müssen menschlichen Ursprungs sein.

---

## ⚧ 7. Gendersensible Sprache

Die FH OÖ schreibt gendersensible Sprache zwingend vor (fließt in die Beurteilung ein).
- **Varianten:** Bevorzugt wird der **Genderstern (`*`)** (z.B. `Professor*innen`).
- **Alternative:** Geschlechtsneutrale Formulierungen (z.B. `Studierende`, `Lehrkräfte`).

---

## 📝 8. Layout und Formatierung (LaTeX / Word)

- Bevorzugt LaTeX verwenden (MTD-Template).
- Rand: 2cm bis 2.5cm.
- A4 Format.
- Tabellen: Niemals als Bild einfügen! Keine vertikalen Linien.
- Typografie: Vermeidung von Waisenkindern und Witwen.
- Abbildungen und Tabellen müssen immer im Text referenziert werden.
- Warnungen am Ende beheben: Vor der finalen Abgabe müssen alle LaTeX-Warnungen bereinigt sein.

---
> ENDE DES REGELWERKS
