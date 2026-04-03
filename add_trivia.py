import os
import glob
import re

TRIVIA = {
    "1_🔭_Solar_Timeseries.py": "st.info(\"💡 **Historical Trivia:** The **Maunder Minimum** (1645–1715) was a period when sunspots became exceedingly rare. This era coincided with the 'Little Ice Age' in Europe and North America. Conversely, the highest explicitly recorded sunspot number occurred during Solar Cycle 19 in 1957 (SSN peaking over 350).\")",
    "2_📊_System_Overview.py": "st.info(\"💡 **Historical Trivia:** The largest recorded cascade of solar energy hitting Earth was the **Carrington Event (1859)**. A massive CME impacted the magnetosphere, causing telegraph lines to spark and catch fire as geomagnetically induced currents surged through the copper wires.\")",
    "4_🌡️_Storm_Simulator.py": "st.info(\"💡 **Historical Trivia:** In March 1989, a severe G5 geomagnetic storm induced enormous electrical currents in the ground, completely melting a massive power transformer in New Jersey and causing a 9-hour system-wide blackout across the entire Hydro-Québec power grid.\")",
    "5_🌍_Geospatial_Impact.py": "st.info(\"💡 **Historical Trivia:** During the 1859 Carrington Event, the auroral oval was pushed so far toward the equator that people in the Caribbean (Cuba, Jamaica) and Hawaii reported seeing the Northern Lights. Gold miners in the Rocky Mountains woke up and began making breakfast, thinking it was morning!\")",
    "6_📅_Monthly_Stats.py": "st.info(\"💡 **Historical Trivia:** Solar cycles average 11 years, but they can vary. Solar Cycle 4 (beginning in 1784) lasted 13.6 years, while Solar Cycle 2 (1766) was only 9 years long.\")",
    "7_📈_Data_Smoothing.py": "st.info(\"💡 **Historical Trivia:** Observational noise in early historical SSN data often came from atmospheric cloud cover over observatories or variations in the telescope optics of the 18th and 19th centuries, necessitating algorithmic smoothing to find true cyclic trends.\")",
    "8_🔗_Correlations.py": "st.info(\"💡 **Historical Trivia:** In 1946, Arthur Covington verified the strict correlation between the F10.7 cm radio flux and Sunspot Number using a repurposed military radar, proving that you could 'listen' to solar activity even on cloudy days.\")",
    "9_⏱️_Lag_Analysis.py": "st.info(\"💡 **Historical Trivia:** While CMEs usually take 2 to 4 days to reach Earth, the August 1972 solar storm CME made the transit in a record-breaking **~14.6 hours**. The resulting magnetic fluctuations triggered dozens of magnetic sea mines along the coast of Vietnam.\")",
    "10_🌊_Periodicity.py": "st.info(\"💡 **Historical Trivia:** The ~11-year periodicity was discovered by amateur astronomer **Heinrich Schwabe** in 1843. He spent 17 years looking for a hypothetical planet inside Mercury's orbit by tracking sunspots, and accidentally discovered the solar cycle instead!\")",
    "11_🔄_Phase_Climatology.py": "st.info(\"💡 **Historical Trivia:** Some of the most devastating storms occur *after* Solar Maximum. The infamous Halloween Storms of 2003 happened 3.5 years *after* the peak of Solar Cycle 23, deep into the declining phase.\")",
    "12_↔️_Hysteresis.py": "st.info(\"💡 **Historical Trivia:** The hysteresis effect implies delayed consequences. It explains why satellite operators experience worse atmospheric drag anomalies years after the sunspot peak has passed, catching unprepared orbital trajectory calculations off guard.\")",
    "13_⚠️_Extreme_Events.py": "st.info(\"💡 **Historical Trivia:** The **Halloween Storms (Oct-Nov 2003)** were so intense (Kp 9, Dst -383 nT) that they shut down the MARTIE martian radiation experiment in orbit around Mars, forced aircraft to re-route globally, and damaged 28 satellites.\")"
}

for filename, trivia_content in TRIVIA.items():
    filepath = f"pages/{filename}"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            content = f.read()
        
        if "Historical Trivia" not in content:
            # We want to inject it after the expander for "What We Are Analyzing"
            pattern = re.compile(r'(with st\.expander\(".*?What We Are Analyzing.*?", expanded=False\):\s+st\.markdown\(""".*?"""\)|with st\.expander\(".*?What We Are Analyzing.*?", expanded=False\):\s+st\.markdown\(\'\'\'.*?\'\'\'\))\s*\n', re.DOTALL)
            
            match = pattern.search(content)
            if match:
                injection = f"{match.group(0)}\n{trivia_content}\n\n"
                new_content = content.replace(match.group(0), injection)
                with open(filepath, "w") as f:
                    f.write(new_content)
                print(f"Added trivia to {filename}")
            else:
                print(f"Could not find injection point in {filename}")
        else:
            print(f"Trivia already exists in {filename}")
