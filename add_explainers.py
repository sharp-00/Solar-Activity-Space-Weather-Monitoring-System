import os
import glob

EXPLAINERS = {
    "1_🔭_Solar_Timeseries.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** We plot decades of historical sunspot, radio flux, and flare data on a continuous timeline. 
    
    **Goal:** To physically visualize the ~11-year solar cycle (Schwabe cycle) across multiple generations. By comparing the crests and troughs (Solar Maximums and Minimums), we identify historical patterns in solar volatility.
    ''')
""",
    "2_📊_System_Overview.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** We map cause (solar eruptions, F10.7, SSN) and effect (Kp index, Dst drop) concurrently.
    
    **Goal:** To provide a single unified executive dashboard showing the entire cascade of energy transfer from the Sun to the Earth's magnetosphere. This is where you see the holistic "weather forecast" taking place in real time or selected historical intervals.
    ''')
""",
    "4_🌡️_Storm_Simulator.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** A simulated environment testing different threshold bounds of geomagnetic storms categorized from NOAA G1 (minor) to G5 (extreme).
    
    **Goal:** To translate abstract physical units (like $nT$ or $Kp$ numbers) into understandable real-world socio-economic impacts. It maps the severity indices directly to threats against satellite orbits, power grids, and aviation communications.
    ''')
""",
    "5_🌍_Geospatial_Impact.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** We calculate and project the "auroral oval" onto an Earth map using historical / simulated Kp numbers.
    
    **Goal:** To visually answer the question "How far south (or north) will the aurora be visible?". Severe geomagnetic storms widen the auroral footprint, drastically expanding the latitude range of visible Northern/Southern lights and mapping geomagnetically induced current (GIC) risk.
    ''')
""",
    "6_📅_Monthly_Stats.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Re-sampling and aggregating high-frequency daily telemetry into monthly bins.
    
    **Goal:** To strip away short-term perturbations (like solar rotation or singular flares) to reveal the majestic, long-lasting climatic shifts of the solar cycle.
    ''')
""",
    "7_📈_Data_Smoothing.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Applying different mathematical filters (e.g., Simple Moving Averages, Savitzky-Golay) to raw telemetry signals.
    
    **Goal:** Real-world data is inherently noisy due to ground sensor fluctuations, instrument switching, and the Sun's 27-day axial rotation. We attempt to discover the "true baseline" by filtering out this high-frequency noise without losing the signal of authentic anomalies.
    ''')
""",
    "8_🔗_Correlations.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Computing the Pearson Correlation Coefficients ($r$) between the various measured indices to construct a correlation matrix.
    
    **Goal:** Looking for collinearity. We want to statistically prove that as Sunspot Number goes up, the Solar Radio Flux ($F10.7$) reliably goes up with it, and that high flares correlate to subsequent geomagnetic disturbances.
    ''')
""",
    "9_⏱️_Lag_Analysis.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Calculating cross-correlations by shifting one time-series forward or backward by "lags" (ranging from -30 to +30 days).
    
    **Goal:** Space is vast. Light (causing $F10.7$ and X-ray flares) takes 8.3 minutes to reach Earth, while plasma (Coronal Mass Ejections causing $Kp$ spikes and $Dst$ drops) takes anywhere from 1 to 4 days. Lag Analysis mathematically measures the exact "transit time" of these particle streams.
    ''')
""",
    "10_🌊_Periodicity.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Moving from the time domain into the frequency domain using Fast Fourier Transforms (FFT) and Continuous Wavelet Transforms (CWT, usually Morlet wavelets).
    
    **Goal:** Instead of just "eyeballing" the 11-year cycle, this rigorously extracts the dominant spectral power peaks. It proves mathematically that a consistent period exists, and identifies secondary harmonic periods (like the ~27 day solar rotation).
    ''')
""",
    "11_🔄_Phase_Climatology.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Normalizing time into "fractions of a solar cycle" (where $0$ is the start of a cycle and $1.0$ is the end) and binning storm occurrences into these fractional buckets.
    
    **Goal:** To map the "danger zone". It answers exactly *where* in the 11-year calendar geomagnetic storms are most likely. It turns out storms do not always happen exactly at Solar Maximum!
    ''')
""",
    "12_↔️_Hysteresis.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Plotting Solar Activity (SSN) against Geomagnetic Activity (Kp) and colorizing the points based on whether the cycle is actively *rising* to a peak or *falling* away from it.
    
    **Goal:** To expose an asymmetry in space weather: the Earth is battered by more severe storms *after* the solar maximum has passed. During the declining phase, coronal holes migrate towards the solar equator, spewing recurrent high-speed solar wind streams. The sun is calming down, but the storms are getting worse!
    ''')
""",
    "13_⚠️_Extreme_Events.py": """with st.expander("ℹ️ What are we analyzing here?", expanded=False):
    st.markdown('''
    **What we are doing:** Utilizing thresholding and $Z$-score outlier detection ($>2.5\sigma$) to automatically flag massive anomalies in the time-series.
    
    **Goal:** Historical identification. Instead of burying massive events (like the Halloween Solar Storms or the Bastille Day Flare) inside massive statistical averages, we explicitly hunt for them to review their specific, unique kinetic signatures.
    ''')
"""
}

for filename, content in EXPLAINERS.items():
    filepath = f"pages/{filename}"
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            lines = f.readlines()
        
        inject_idx = -1
        # Try to find st.title, st.header, or st.subheader
        for i, line in enumerate(lines):
            if line.strip().startswith("st.title(") or line.strip().startswith("st.header("):
                inject_idx = i + 1
                break
        
        # Filter double injections if we re-run
        if not any("ℹ️ What are we analyzing here?" in l for l in lines):
            if inject_idx != -1:
                lines.insert(inject_idx, f"\n{content}\n")
                with open(filepath, "w") as f:
                    f.writelines(lines)
                print(f"Updated {filename}")
            else:
                print(f"Could not find title in {filename}")
        else:
            print(f"Already updated {filename}")
