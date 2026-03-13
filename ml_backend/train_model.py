"""
TriageDelRo ML Backend — Model Training
Trains Random Forest classifiers for:
  - Category: Flooding | Fire | Medical Emergency | Infrastructure Damage
  - Severity: High | Medium | Low

Run: python train_model.py
"""

import os, pickle, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Dataset ────────────────────────────────────────────────────────────────────
# Filipino/English disaster reports for Barangay Del Rosario, Milaor

SAMPLES = [
    # ── FLOODING ──────────────────────────────────────────────────────────────
    # High
    ("Baha na mataas sa Purok 3, hindi na makalabas ang mga tao", "Flooding", "High"),
    ("Flash flood warning, street completely underwater", "Flooding", "High"),
    ("Families trapped on rooftops due to rising floodwater", "Flooding", "High"),
    ("Malakas na ulan, baha na hanggang dibdib sa Purok 5", "Flooding", "High"),
    ("Flood submerged entire road, vehicles cannot pass, people stranded", "Flooding", "High"),
    ("Rising waters entering homes, 20 families need evacuation", "Flooding", "High"),
    ("Mataas na baha, nakalubog na ang bahay sa Sitio Maligaya", "Flooding", "High"),
    ("Flood level critical, barangay road impassable, elderly trapped", "Flooding", "High"),
    ("Umaapaw na ang ilog, baha papasok sa mga bahay", "Flooding", "High"),
    ("Severe flooding, rescue boats needed at Purok 4", "Flooding", "High"),
    # Medium
    ("Nagbabaha sa kalsada, mababa pero nakakaistorbo sa trapiko", "Flooding", "Medium"),
    ("Street flooding after heavy rain, ankle deep water", "Flooding", "Medium"),
    ("Baha sa kalsada, pwede pa lumakad pero mapanganib na", "Flooding", "Medium"),
    ("Flood water rising slowly, residents on alert", "Flooding", "Medium"),
    ("Low-lying area flooding, some residents preparing to evacuate", "Flooding", "Medium"),
    ("Pagbaha sa purok, moderate level, nakaapekto sa ilang bahay", "Flooding", "Medium"),
    ("Knee-high flooding in the market area, businesses affected", "Flooding", "Medium"),
    ("Patuloy na umaangat ang tubig sa ilang bahagi ng barangay", "Flooding", "Medium"),
    # Low
    ("Slight flooding sa kanto pagkatapos ng ulan, mabilis mawawala", "Flooding", "Low"),
    ("Minor water accumulation on road, draining slowly", "Flooding", "Low"),
    ("Maliit na baha sa pasukan ng barangay, hindi malubha", "Flooding", "Low"),
    ("Small puddles forming after rain, not dangerous yet", "Flooding", "Low"),

    # ── FIRE ──────────────────────────────────────────────────────────────────
    # High
    ("Nasusunog ang tatlong bahay sa Purok 2, tumakas na ang mga residente", "Fire", "High"),
    ("House fire spreading to adjacent structures, multiple families displaced", "Fire", "High"),
    ("Malaking sunog, naglalagablab na ang apartment sa may palengke", "Fire", "High"),
    ("Fire engulfing warehouse, thick smoke visible from barangay", "Fire", "High"),
    ("Sunog sa Purok 7, walang makalapit dahil sa init at usok", "Fire", "High"),
    ("Major fire, 5 houses burning, residents evacuating", "Fire", "High"),
    ("Nasusunog ang LPG store, panganib sa mga kapitbahay", "Fire", "High"),
    ("Structure fire with reported casualties, fire trucks needed", "Fire", "High"),
    # Medium
    ("Sunog sa bakuran, apoy kontrolado pa pero lumalaki", "Fire", "Medium"),
    ("Small house fire, one room affected, neighbors helping", "Fire", "Medium"),
    ("Fire in vacant lot, spreading slowly toward houses", "Fire", "Medium"),
    ("Nasusunog ang isang bahay, napatay na ang isa pero hindi pa ganap", "Fire", "Medium"),
    ("Grass fire near residential area, wind pushing it closer", "Fire", "Medium"),
    # Low
    ("Maliit na sunog sa basurahan, napatay na ng mga residente", "Fire", "Low"),
    ("Minor fire in kitchen, extinguished by homeowner", "Fire", "Low"),
    ("Nasunog ang mga dahon sa bakuran, kontrolado at maliit", "Fire", "Low"),
    ("Small campfire got out of hand, neighbors put it out", "Fire", "Low"),

    # ── MEDICAL EMERGENCY ──────────────────────────────────────────────────────
    # High
    ("May naaksidente sa kalsada, may sugat at walang malay", "Medical Emergency", "High"),
    ("Person collapsed unconscious on the street, not breathing", "Medical Emergency", "High"),
    ("Matanda na nahihirapan huminga, kailangan ng ambulansya agad", "Medical Emergency", "High"),
    ("Bata nalunod sa baha, di pa nakakagising", "Medical Emergency", "High"),
    ("Multiple casualties in road accident, urgent medical attention needed", "Medical Emergency", "High"),
    ("Lalaki tinamaan ng kidlat, di na gumagalaw", "Medical Emergency", "High"),
    ("Heart attack suspected, elderly patient unresponsive", "Medical Emergency", "High"),
    ("Serious injury from falling debris, bleeding heavily", "Medical Emergency", "High"),
    # Medium
    ("May na-food poison sa selebrasyon, ilang tao may sakit tiyan", "Medical Emergency", "Medium"),
    ("Person injured in minor accident, walking but in pain", "Medical Emergency", "Medium"),
    ("Ilang bata may lagnat at ubo, posibleng dengue", "Medical Emergency", "Medium"),
    ("Elderly fell down, possible broken bone but conscious", "Medical Emergency", "Medium"),
    ("Minor laceration from broken glass, needs stitches", "Medical Emergency", "Medium"),
    ("Asthma attack, patient conscious but struggling to breathe", "Medical Emergency", "Medium"),
    # Low
    ("Bata may sakit, kailangan ng gamot pero hindi emergency", "Medical Emergency", "Low"),
    ("Minor cut from accident, needs first aid", "Medical Emergency", "Low"),
    ("Resident needs medical consultation, not urgent", "Medical Emergency", "Low"),
    ("Mild fever, requesting barangay health center assistance", "Medical Emergency", "Low"),

    # ── INFRASTRUCTURE DAMAGE ──────────────────────────────────────────────────
    # High
    ("Bumagsak ang malaking puno sa kalsada, naharang ang daan", "Infrastructure Damage", "High"),
    ("Bridge structurally damaged, dangerous to cross", "Infrastructure Damage", "High"),
    ("Electrical post knocked down by storm, live wires on road", "Infrastructure Damage", "High"),
    ("Malaking butas sa kalsada, nahulog na ang motor", "Infrastructure Damage", "High"),
    ("Retaining wall collapsed, blocking main road to barangay", "Infrastructure Damage", "High"),
    ("Nabasag ang tubería, walang tubig ang buong Purok 6", "Infrastructure Damage", "High"),
    ("Landslide blocking road, several houses at risk", "Infrastructure Damage", "High"),
    ("Downed power lines sparking on flooded road, extreme danger", "Infrastructure Damage", "High"),
    # Medium
    ("Damaged road with large potholes affecting vehicles", "Infrastructure Damage", "Medium"),
    ("Street light broken sa may intersection, mapanganib sa gabi", "Infrastructure Damage", "Medium"),
    ("Nabali ang fence ng barangay hall dahil sa bagyo", "Infrastructure Damage", "Medium"),
    ("Footbridge damaged, residents taking longer route", "Infrastructure Damage", "Medium"),
    ("Clogged drainage causing minor flooding, needs cleaning", "Infrastructure Damage", "Medium"),
    ("Nasirang gate ng covered court, kailangang ayusin", "Infrastructure Damage", "Medium"),
    # Low
    ("Maliit na butas sa kalsada, kailangan ng patching", "Infrastructure Damage", "Low"),
    ("Minor damage to barangay signage after storm", "Infrastructure Damage", "Low"),
    ("Basura nakaharang sa kanal, kailangan i-clear", "Infrastructure Damage", "Low"),
    ("Broken sidewalk tile, minor trip hazard", "Infrastructure Damage", "Low"),
    ("Paint peeling off barangay hall, needs repainting", "Infrastructure Damage", "Low"),

    # ── EXTRA LOW SAMPLES (added to fix class imbalance) ─────────────────────
    # Flooding Low
    ("Mababaw lang ang baha sa kanto, lilipas pagtirik ng araw", "Flooding", "Low"),
    ("Slight water on the road after rain, passable", "Flooding", "Low"),
    ("Konting tubig sa bangketa, hindi nakaka-abala", "Flooding", "Low"),
    ("Water pooling near the drainage but not overflowing", "Flooding", "Low"),
    ("Light rain causing small puddles, residents not alarmed", "Flooding", "Low"),
    # Fire Low
    ("Burning dried leaves in the yard, owner has it under control", "Fire", "Low"),
    ("Maliit na apoy mula sa upos ng sigarilyo, napatay na agad", "Fire", "Low"),
    ("Candle caused a small scorch on curtain, fire out now", "Fire", "Low"),
    # Medical Low
    ("Naapakan ng pako, kailangan ng tetanus shot", "Medical Emergency", "Low"),
    ("Minor allergic reaction, patient stable and conscious", "Medical Emergency", "Low"),
    ("Headache and dizziness, requesting barangay health visit", "Medical Emergency", "Low"),
    ("Bata may ubo at sipon, gusto ng gamot mula sa health center", "Medical Emergency", "Low"),
    # Infrastructure Low
    ("Nabali ang sanga ng puno sa tapat ng bahay, kailangan i-trim", "Infrastructure Damage", "Low"),
    ("Faded road markings near school, needs repainting", "Infrastructure Damage", "Low"),
    ("Loose manhole cover on the sidewalk, minor hazard", "Infrastructure Damage", "Low"),
    ("Cracked concrete near the plaza, no one has fallen yet", "Infrastructure Damage", "Low"),

    # ── EXTRA MEDIUM SAMPLES (added to fix class imbalance) ──────────────────
    # Flooding Medium
    ("Baha sa kalye, halos tuhod na, ilang bahay apektado", "Flooding", "Medium"),
    ("Moderate flooding in the market area, vendors leaving", "Flooding", "Medium"),
    ("Water level rising in low-lying areas, BDRRMC monitoring", "Flooding", "Medium"),
    # Fire Medium
    ("Small structure fire, one family affected, neighbors alert", "Fire", "Medium"),
    ("Fire in the backyard spreading to fence, residents watching", "Fire", "Medium"),
    ("Sunog sa tindahan, hindi pa malaki pero lumalaki ang usok", "Fire", "Medium"),
    # Medical Medium
    ("Two residents with high fever after flood, needs check-up", "Medical Emergency", "Medium"),
    ("Bata nahulog mula sa puno, masakit ang braso, conscious", "Medical Emergency", "Medium"),
    ("Minor vehicular accident, passenger has head bruise", "Medical Emergency", "Medium"),
    # Infrastructure Medium
    ("Butas sa kalsada, malaking sasakyan nabaon na doon", "Infrastructure Damage", "Medium"),
    ("Broken streetlight at intersection, dangerous at night", "Infrastructure Damage", "Medium"),
    ("Water pipe leaking, affecting water supply in 3 houses", "Infrastructure Damage", "Medium"),

    # ── EXTRA MIXED SAMPLES ───────────────────────────────────────────────────
    ("Tumataas ang tubig sa buong barangay, baha na", "Flooding", "High"),
    ("Sunog sa kabahayan, tumakas na ang pamilya", "Fire", "High"),
    ("Nasaktan sa gusaling bumagsak, maraming sugatan", "Medical Emergency", "High"),
    ("Bumagsak na haligi sa gitna ng kalsada", "Infrastructure Damage", "High"),
    ("Moderate flooding sa ilang lugar, nagbababala ang BDRRMC", "Flooding", "Medium"),
    ("Kitchen fire, homeowner trying to extinguish", "Fire", "Medium"),
    ("Injured resident needs help getting to hospital", "Medical Emergency", "Medium"),
    ("Road with cracks, dangerous for motorcycles", "Infrastructure Damage", "Medium"),
    ("Slight tubig sa kalsada pagkatapos ng ulan", "Flooding", "Low"),
    ("Maliit na apoy sa bakuran, napatay na ng kapitbahay", "Fire", "Low"),
    ("Bata may sakit, requesting health worker visit", "Medical Emergency", "Low"),
    ("Damaged drainage cover, minor safety concern", "Infrastructure Damage", "Low"),
    ("Flash flood warning issued, rivers overflowing", "Flooding", "High"),
    ("Sunog sa bodega, naglalagablab na", "Fire", "High"),
    ("Elderly person collapsed, unresponsive", "Medical Emergency", "High"),
    ("Electrical wire down on road after typhoon", "Infrastructure Damage", "High"),
    ("Bahang nakaapekto sa daan, mababa ang tubig", "Flooding", "Medium"),
    ("Grass fire moving toward houses", "Fire", "Medium"),
    ("Several residents vomiting after community meal", "Medical Emergency", "Medium"),
    ("Pothole causing accidents, needs immediate repair", "Infrastructure Damage", "Medium"),
    ("Konting baha sa kanto, lilipas din", "Flooding", "Low"),
    ("Small fire from burning trash, controlled", "Fire", "Low"),
    ("Minor bruise from fall, needs first aid kit", "Medical Emergency", "Low"),
    ("Missing road sign near school zone", "Infrastructure Damage", "Low"),
    ("Walang makalabas dahil sa mataas na baha sa gate", "Flooding", "High"),
    ("Nasabog ang kalan, may sunog sa bahay", "Fire", "High"),
    ("Naaksidente ang tricycle, driver hindi makakilos", "Medical Emergency", "High"),
    ("Nabasag ang tubo ng tubig, bumubuhos na tubig sa kalsada", "Infrastructure Damage", "High"),
    ("Baha na sa ibabang bahagi ng barangay", "Flooding", "Medium"),
    ("Di pa malaking sunog pero lumalaki", "Fire", "Medium"),
    ("Nahulog mula sa puno, may bali", "Medical Emergency", "Medium"),
    ("Nasirang kuweberta, mapanganib sa mga bata", "Infrastructure Damage", "Medium"),
    ("May konting tubig sa likod ng bahay, hindi pa malubha", "Flooding", "Low"),
    ("Napaso ng mainit na pagkain, minor burn", "Medical Emergency", "Low"),
    ("Nabali ang kawayan na bakod, kailangan ayusin", "Infrastructure Damage", "Low"),
]

def main():
    texts = [s[0] for s in SAMPLES]
    categories = [s[1] for s in SAMPLES]
    severities = [s[2] for s in SAMPLES]

    # ── Category model ─────────────────────────────────────────────────────
    cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            strip_accents='unicode',
        )),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        texts, categories, test_size=0.2, random_state=42, stratify=categories
    )
    cat_pipeline.fit(X_train, y_train)
    y_pred = cat_pipeline.predict(X_test)
    print("\n=== CATEGORY MODEL ===")
    print(classification_report(y_test, y_pred))

    # ── Severity model ─────────────────────────────────────────────────────
    sev_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            strip_accents='unicode',
        )),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=12,          # prevent overfitting
            min_samples_leaf=2,    # prevent overfitting on small classes
            class_weight='balanced',  # FIX: compensates for High/Medium/Low imbalance
            random_state=42,
            n_jobs=-1,
        )),
    ])

    X_train2, X_test2, y_train2, y_test2 = train_test_split(
        texts, severities, test_size=0.2, random_state=42, stratify=severities
    )
    sev_pipeline.fit(X_train2, y_train2)
    y_pred2 = sev_pipeline.predict(X_test2)
    print("\n=== SEVERITY MODEL ===")
    print(classification_report(y_test2, y_pred2))

    # ── Save models ────────────────────────────────────────────────────────
    os.makedirs('models', exist_ok=True)
    with open('models/category_model.pkl', 'wb') as f:
        pickle.dump(cat_pipeline, f)
    with open('models/severity_model.pkl', 'wb') as f:
        pickle.dump(sev_pipeline, f)

    # Save labels for reference
    labels = {
        'categories': sorted(set(categories)),
        'severities': sorted(set(severities)),
    }
    with open('models/labels.json', 'w') as f:
        json.dump(labels, f, indent=2)

    print("\n✅ Models saved to models/")
    print(f"   Categories: {labels['categories']}")
    print(f"   Severities: {labels['severities']}")

if __name__ == '__main__':
    main()
