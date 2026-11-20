import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import folium
from streamlit_folium import st_folium


st.set_page_config(page_title="AGRO-SENTINEL", page_icon="🌱", layout="wide")

DATA_FILE = Path("agro_sentinel_readings.csv")
OPEN_METEO_HOURLY_FIELDS = (
    "temperature_2m,relative_humidity_2m,shortwave_radiation,direct_radiation,"
    "diffuse_radiation,direct_normal_irradiance,cloud_cover,precipitation,wind_speed_10m"
)


@st.cache_data(ttl=3600)
def fetch_nasa_power_hourly(latitude, longitude, selected_date):
    """Fetch one day's NASA POWER hourly meteorology and satellite-derived radiation."""
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "T2M,RH2M,ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_DIFF,ALLSKY_SFC_SW_DNI",
        "community": "AG",
        "longitude": longitude,
        "latitude": latitude,
        "start": selected_date.strftime("%Y%m%d"),
        "end": selected_date.strftime("%Y%m%d"),
        "format": "JSON",
        "time-standard": "UTC",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    parameters = payload.get("properties", {}).get("parameter", {})
    if not parameters.get("T2M") or not parameters.get("RH2M"):
        raise ValueError(
            "NASA POWER has no valid values for that date and location. "
            "Try an earlier date; recent data may not be available yet."
        )
    rows = []
    for hour, temperature in parameters["T2M"].items():
        humidity = parameters["RH2M"].get(hour, -999)
        if temperature == -999 or humidity == -999:
            continue
        rows.append(
            {
                "time": datetime.strptime(hour, "%Y%m%d%H").replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Africa/Kampala")),
                "temperature_c": float(temperature),
                "relative_humidity_pct": float(humidity),
                "solar_mj_m2_per_hour": _nasa_value(parameters, "ALLSKY_SFC_SW_DWN", hour),
                "diffuse_solar_mj_m2_per_hour": _nasa_value(parameters, "ALLSKY_SFC_SW_DIFF", hour),
                "direct_normal_mj_m2_per_hour": _nasa_value(parameters, "ALLSKY_SFC_SW_DNI", hour),
            }
        )
    if not rows:
        raise ValueError("NASA POWER returned no usable hourly values for this date.")
    return pd.DataFrame(rows)


def _nasa_value(parameters, parameter_name, hour):
    value = parameters.get(parameter_name, {}).get(hour, -999)
    return None if value == -999 else float(value)


@st.cache_data(ttl=60)
def fetch_open_meteo_forecast(latitude, longitude):
    """Fetch seven days of hourly forecast estimates for a point in Uganda."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": OPEN_METEO_HOURLY_FIELDS,
        "forecast_days": 7,
        "timezone": "Africa/Kampala",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    forecast = _open_meteo_frame(hourly)
    if forecast.empty:
        raise ValueError("The forecast service returned no usable hourly values.")
    forecast["time"] = forecast["time"].dt.floor("h")
    aggregations = {column: "mean" for column in forecast.columns if column != "time"}
    if "precipitation" in aggregations:
        aggregations["precipitation"] = "sum"
    forecast = forecast.groupby("time", as_index=False).agg(aggregations)
    offset = int(payload.get("utc_offset_seconds", 10800))
    now_uganda = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=offset)
    next_full_hour = pd.Timestamp(now_uganda).ceil("h")
    upcoming = forecast.loc[forecast["time"] >= next_full_hour].reset_index(drop=True)
    if upcoming.empty:
        raise ValueError("No future forecast hours were returned. Try again shortly.")
    return upcoming, now_uganda


@st.cache_data(ttl=3600)
def fetch_open_meteo_archive(latitude, longitude, start_date, end_date):
    """Fetch past hourly reanalysis estimates; recent dates may not be archived yet."""
    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": OPEN_METEO_HOURLY_FIELDS,
            "timezone": "Africa/Kampala",
        },
        timeout=30,
    )
    response.raise_for_status()
    historical = _open_meteo_frame(response.json().get("hourly", {}))
    if historical.empty:
        raise ValueError("No historical hourly data was returned for that date range.")
    return historical


def _open_meteo_frame(hourly):
    if "time" not in hourly:
        raise ValueError("The weather response did not contain an hourly time series.")
    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"], errors="coerce")})
    for field in OPEN_METEO_HOURLY_FIELDS.split(","):
        if field in hourly:
            frame[field] = pd.to_numeric(hourly[field], errors="coerce")
    frame = frame.dropna(subset=["time", "temperature_2m", "relative_humidity_2m"])
    return frame.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relative_humidity_2m": "relative_humidity_pct",
        }
    )


def show_weather_charts(frame):
    """Plot forecast/archive variables in separate unit-compatible groups."""
    if "temperature_c" in frame and "relative_humidity_pct" in frame:
        st.markdown("**Air temperature (°C)**")
        st.line_chart(frame.set_index("time")[["temperature_c"]])
        st.markdown("**Relative humidity (%)**")
        st.line_chart(frame.set_index("time")[["relative_humidity_pct"]])
    radiation_fields = [
        field for field in ("shortwave_radiation", "direct_radiation", "diffuse_radiation")
        if field in frame
    ]
    if radiation_fields:
        st.markdown("**Solar radiation (W/m²)**")
        st.line_chart(frame.set_index("time")[radiation_fields])
    other_fields = [field for field in ("cloud_cover", "precipitation", "wind_speed_10m") if field in frame]
    if other_fields:
        st.markdown("**Cloud cover, precipitation and wind (their documented units)**")
        st.line_chart(frame.set_index("time")[other_fields])


@st.cache_data(ttl=900)
def fetch_forecast_grid(latitude, longitude, spacing_km, forecast_time):
    """Fetch one forecast hour for nine nearby grid points for a simple GIS view."""
    lat_step = spacing_km / 111.0
    lon_step = spacing_km / (111.0 * max(abs(math.cos(math.radians(latitude))), 0.2))
    points = [
        (latitude + row * lat_step, longitude + col * lon_step)
        for row in (-1, 0, 1)
        for col in (-1, 0, 1)
    ]
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": ",".join(f"{lat:.5f}" for lat, _ in points),
            "longitude": ",".join(f"{lon:.5f}" for _, lon in points),
            "hourly": "temperature_2m,relative_humidity_2m",
            "forecast_days": 2,
            "timezone": "Africa/Kampala",
        },
        timeout=30,
    )
    response.raise_for_status()
    locations = response.json()
    if isinstance(locations, dict):
        locations = [locations]
    rows = []
    for location in locations:
        hourly = location.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []), errors="coerce")
        temperatures = hourly.get("temperature_2m", [])
        humidities = hourly.get("relative_humidity_2m", [])
        future_indices = [i for i, time in enumerate(times) if pd.notna(time) and time >= forecast_time]
        if not future_indices:
            continue
        index = future_indices[0]
        temp = temperatures[index]
        humidity = humidities[index]
        if temp is None or humidity is None:
            continue
        if temp >= 38 or (temp >= 35 and humidity >= 70):
            category, color = "HIGH (prototype rule)", "#d62728"
        elif temp >= 32 or (temp >= 29 and humidity >= 75):
            category, color = "ELEVATED (prototype rule)", "#ff7f0e"
        else:
            category, color = "LOWER (prototype rule)", "#2ca02c"
        rows.append(
            {
                "lat": location["latitude"],
                "lon": location["longitude"],
                "temperature_c": float(temp),
                "humidity_pct": float(humidity),
                "category": category,
                "risk_color": color,
            }
        )
    if not rows:
        raise ValueError("No nearby forecast points were available to draw the GIS estimate.")
    return pd.DataFrame(rows)

operator_requested = "operator" in st.query_params
is_public_view = not operator_requested

if operator_requested:
    try:
        operator_password = st.secrets["operator_password"]
    except Exception:
        operator_password = None

    if operator_password:
        entered_password = st.sidebar.text_input(
            "Operator password",
            type="password",
            key="operator_password_entry",
        )
        is_public_view = entered_password != operator_password
    else:
        is_public_view = True
        st.sidebar.warning("Operator access is not configured. Add operator_password to Streamlit secrets.")

if is_public_view:
    st.title("🌱 Local Weather & Heat Preparedness")
    st.caption("A simple local forecast and weather-history view for planning and awareness.")
    st.warning(
        "Forecasts describe possible weather; they do not predict whether a person will become ill. "
        "This demonstration is not an official warning service or a safety clearance. Follow local "
        "health and workplace guidance, and use appropriate on-site measurements."
    )
else:
    st.title("🌱 AGRO-SENTINEL — Project Operator")
    st.caption("Environmental data demonstrator, forecast explorer and station-management view")
    st.warning(
        "DEMO MODE: No physical sensors are connected. Forecasts and gridded estimates are not live "
        "measurements. The example screening rules have not been validated for medical or occupational decisions."
    )

app_time = datetime.now(ZoneInfo("Africa/Kampala")).strftime("%A, %d %B %Y — %H:%M EAT")

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; }
    .agro-hero {
        background: linear-gradient(135deg, #effaf2 0%, #ffffff 70%, #fff8dc 100%);
        border: 1px solid #90be96;
        border-radius: 18px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .agro-hero h2 { color: #1d4d25; margin-bottom: 0.25rem; }
    .agro-hero p { color: #263238; }
    </style>
    <div class="agro-hero">
      <h2>Protect the people who feed us</h2>
      <p>Solar-powered environmental monitoring can help field teams recognize heat exposure, plan safer breaks and respond earlier.</p>
    </div>
    <p><strong>App time:</strong> __APP_TIME__</p>
    """.replace("__APP_TIME__", app_time),
    unsafe_allow_html=True,
)

with st.expander("🌾 Why heat and sunlight preparedness matters", expanded=False):
    farmer_col, solar_col = st.columns(2)
    with farmer_col:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/d/d5/Love_and_family_at_work_in_Uganda.jpg",
            caption="Agricultural work in Uganda — representative open-access photo",
            use_container_width=True,
        )
    with solar_col:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/8/81/Innovation_Africa_-_solar_panels.jpg",
            caption="Solar power can support remote monitoring in areas without grid power",
            use_container_width=True,
        )
    st.markdown(
        "**Solar-powered AGRO-SHIELD:** solar panels charge a battery, which can power a temperature and humidity "
        "sensor, an ESP32 controller and wireless communication. The station can report measurements, while the app "
        "converts the readings into explainable risk categories and break-planning prompts."
    )
    st.markdown(
        "**Sun and skin-health protection:** shade, protective clothing and time away from direct sunlight can reduce "
        "unprotected ultraviolet (UV) exposure. Reducing severe or repeated UV exposure over a lifetime may help lower "
        "skin-cancer risk, but this app cannot remove cancerous cells or claim that sunlight monitoring itself prevents cancer. "
        "Seek medical advice for suspicious skin changes."
    )

if not is_public_view:
    with st.expander("🎬 Project video", expanded=False):
        video_url = st.text_input(
            "Hosted video URL",
            placeholder="Paste a YouTube or direct MP4 URL",
            key="agro_video_url",
            help="Use a YouTube link or an MP4 hosted through GitHub Releases or cloud storage.",
        )
        if video_url:
            st.video(video_url)
        else:
            st.info("Add the project video here after hosting or uploading it.")

if not is_public_view:
    with st.expander("🎤 2:50 Project presentation script", expanded=False):
        st.markdown("### AGRO-SENTINEL: Protecting Agricultural Workers Through Predictive Heat-Health Surveillance")
        st.markdown(
            "**Opening — 0:00–0:25:** Imagine a farmer working under the scorching sun. Rising temperature and humidity "
            "can increase heat stress, while the worker may have no reliable warning or break guidance.\n\n"
            "**Problem and solution — 0:25–0:55:** AGRO-SENTINEL is a proposed digital worker-safety and environmental "
            "monitoring platform for agriculture. AGRO-SHIELD is intended to collect temperature and humidity through solar-powered sensors.\n\n"
            "**Capabilities — 0:55–1:35:** explainable heat-risk assessment, practical break-planning prompts and GIS awareness.\n\n"
            "**Demonstration — 1:35–2:05:** enter or retrieve readings and show how the platform explains the resulting category and safety prompts.\n\n"
            "**Impact — 2:05–2:35:** combine monitoring, explainable assessment and worker-safety planning without claiming validated medical prediction.\n\n"
            "**Closing — 2:35–2:50:** We are Peter Lokong and Yusuf. Protecting agricultural productivity begins with protecting the people who make it possible."
        )

with st.sidebar:
    st.header("Monitoring station")
    st.caption("Choose a location and weather information type.")
    if is_public_view:
        if "site_lat" not in st.session_state:
            st.session_state.site_lat = 0.3476
        if "site_lon" not in st.session_state:
            st.session_state.site_lon = 32.5825
        if "site_name" not in st.session_state:
            st.session_state.site_name = "Kampala, Uganda"
        with st.form("public_location_search"):
            place_query = st.text_input("Find your town or district", value=st.session_state.site_name)
            location_submitted = st.form_submit_button("Find location")
        if location_submitted:
            try:
                location_response = requests.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": place_query, "count": 1, "language": "en", "format": "json"},
                    timeout=20,
                )
                location_response.raise_for_status()
                location_results = location_response.json().get("results", [])
                if not location_results:
                    st.sidebar.error("No matching place found. Try a nearby town or district name.")
                else:
                    match = location_results[0]
                    st.session_state.site_lat = float(match["latitude"])
                    st.session_state.site_lon = float(match["longitude"])
                    st.session_state.site_name = ", ".join(
                        part for part in (match.get("name"), match.get("admin1"), match.get("country")) if part
                    )
            except (requests.RequestException, ValueError, KeyError) as error:
                st.sidebar.error(f"Could not look up that place: {error}")
        station_name = st.session_state.site_name
        public_source = st.selectbox(
            "Weather information",
            ["Next 7-day forecast", "Past hourly history", "NASA radiation history"],
        )
        data_mode = {
            "Next 7-day forecast": "7-day weather forecast (Open-Meteo)",
            "Past hourly history": "Past hourly weather (Open-Meteo archive)",
            "NASA radiation history": "NASA POWER hourly estimate",
        }[public_source]
        with st.expander("Advanced: adjust the map location"):
            latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, format="%.5f", key="site_lat")
            longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, format="%.5f", key="site_lon")
    else:
        data_mode = st.selectbox(
            "Data source",
            [
                "Manual demonstration input",
                "Generate sample reading",
                "NASA POWER hourly estimate",
                "7-day weather forecast (Open-Meteo)",
                "Past hourly weather (Open-Meteo archive)",
                "Upload station CSV",
            ],
        )
        station_name = st.text_input("Station label", "AGRO-SHIELD Demo Station")
        latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=0.3476, format="%.5f")
        longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=32.5825, format="%.5f")
        st.caption("The map marker and forecast location follow these coordinates.")
    if data_mode == "7-day weather forecast (Open-Meteo)" and st.button("Update forecast now"):
        fetch_open_meteo_forecast.clear()

if is_public_view:
    st.subheader("1. Local weather information")
else:
    st.subheader("1. Environmental measurements")
observation_time = datetime.now().astimezone()
source_label = "manual demonstration input"
historical_data = None
if data_mode == "Manual demonstration input":
    c1, c2, c3 = st.columns(3)
    with c1:
        temp = st.number_input("Air temperature (°C)", min_value=-10.0, max_value=60.0, value=32.0, step=0.5)
    with c2:
        humidity = st.number_input("Relative humidity (%)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
    with c3:
        workload = st.selectbox("Work intensity", ["Light", "Moderate", "Heavy"])
else:
    workload = "Moderate" if is_public_view else st.selectbox("Work intensity", ["Light", "Moderate", "Heavy"], index=1)
    if data_mode == "Generate sample reading":
        import random
        temp = round(random.uniform(27, 39), 1)
        humidity = round(random.uniform(40, 90), 1)
        source_label = "generated sample (not a sensor reading)"
        st.info("A random sample has been generated for interface testing. It is not a sensor reading.")
    elif data_mode == "NASA POWER hourly estimate":
        selected_date = st.date_input(
            "NASA POWER date (satellite-derived radiation and weather)",
            value=datetime.now().date() - timedelta(days=365),
            max_value=datetime.now().date(),
        )
        try:
            historical_data = fetch_nasa_power_hourly(latitude, longitude, selected_date)
        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(f"Could not retrieve NASA POWER data: {error}")
            st.stop()
        latest = historical_data.iloc[-1]
        temp = float(latest["temperature_c"])
        humidity = float(latest["relative_humidity_pct"])
        observation_time = latest["time"].to_pydatetime()
        source_label = "NASA POWER gridded hourly estimate"
        st.warning(
            "NASA POWER provides gridded historical/near-real-time estimates. Its surface solar radiation "
            "products are satellite-derived, but this is not live satellite imagery, a local station "
            "measurement, or a real-time worker-safety instrument. Recent values can be delayed."
        )
        st.caption(f"NASA POWER hourly series for {selected_date:%Y-%m-%d}; time shown in Uganda local time (EAT).")
        radiation_data = historical_data.dropna(subset=["solar_mj_m2_per_hour", "diffuse_solar_mj_m2_per_hour"])
        if not radiation_data.empty and radiation_data["solar_mj_m2_per_hour"].sum() > 0:
            st.markdown("**NASA POWER hourly weather and satellite-derived solar radiation**")
            st.markdown("Temperature is °C; humidity is %.")
            st.line_chart(historical_data.set_index("time")[["temperature_c", "relative_humidity_pct"]])
            solar_chart = radiation_data.set_index("time")[["solar_mj_m2_per_hour", "diffuse_solar_mj_m2_per_hour"]]
            st.line_chart(solar_chart)
            total_solar = radiation_data["solar_mj_m2_per_hour"].sum()
            total_diffuse = radiation_data["diffuse_solar_mj_m2_per_hour"].sum()
            direct_horizontal = max(total_solar - total_diffuse, 0)
            if direct_horizontal + total_diffuse > 0:
                solar_mix = pd.DataFrame(
                    {"Component": ["Estimated direct horizontal", "Diffuse"], "Energy (MJ/m²)": [direct_horizontal, total_diffuse]}
                )
                pie = px.pie(solar_mix, names="Component", values="Energy (MJ/m²)", title="Daily solar-energy components (estimate)")
                st.plotly_chart(pie, width="stretch")
            with st.expander("NASA POWER hourly data table"):
                st.dataframe(historical_data, hide_index=True, width="stretch")
        else:
            st.info("NASA returned weather values, but solar radiation values for this date are missing or unavailable.")
    elif data_mode == "7-day weather forecast (Open-Meteo)":
        try:
            forecast, now_uganda = fetch_open_meteo_forecast(latitude, longitude)
        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(f"Could not retrieve the weather forecast: {error}")
            st.stop()

        next_hour = forecast.iloc[0]
        temp = float(next_hour["temperature_c"])
        humidity = float(next_hour["relative_humidity_pct"])
        observation_time = next_hour["time"].to_pydatetime()
        source_label = "Open-Meteo seven-day forecast estimate"
        st.warning(
            "Forecast only: these are model estimates, not observations or incident predictions. "
            "Use forecasts to plan extra checks; confirm conditions with on-site instruments and "
            "follow official occupational-safety guidance."
        )

        st.markdown("**Next three hourly forecast values (Uganda local time, EAT)**")
        next_hours = forecast.head(3).copy()
        next_hours["time"] = next_hours["time"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(next_hours, hide_index=True, width="stretch")

        def forecast_screening(row):
            forecast_temp = row["temperature_c"]
            forecast_humidity = row["relative_humidity_pct"]
            if forecast_temp >= 38 or (forecast_temp >= 35 and forecast_humidity >= 70):
                return "HIGH (prototype rule)"
            if forecast_temp >= 32 or (forecast_temp >= 29 and forecast_humidity >= 75):
                return "ELEVATED (prototype rule)"
            return "BELOW EXAMPLE TRIGGERS (not a safety clearance)"

        if not is_public_view:
            upcoming_24h = forecast.loc[
                forecast["time"] <= now_uganda + timedelta(hours=24)
            ].copy()
            upcoming_24h["screening_category"] = upcoming_24h.apply(forecast_screening, axis=1)
            flagged = upcoming_24h.loc[
                upcoming_24h["screening_category"].str.startswith(("HIGH", "ELEVATED"))
            ]
            if not flagged.empty:
                first_flag = flagged.iloc[0]
                lead_hours = max(0, int((first_flag["time"] - now_uganda).total_seconds() // 3600))
                if lead_hours <= 2:
                    st.error(
                        f"Early heads-up: forecast conditions meet this prototype's "
                        f"{first_flag['screening_category'].split()[0]} screening rule at "
                        f"{first_flag['time']:%Y-%m-%d %H:%M} EAT (about {lead_hours} hours from now). "
                        "This is not a prediction of an incident or a validated safety alert."
                    )
                else:
                    st.info(
                        f"Planning notice: the first forecast hour meeting this prototype's "
                        f"elevated screening rule is {first_flag['time']:%Y-%m-%d %H:%M} EAT "
                        f"(about {lead_hours} hours from now). This does not predict an incident."
                    )
            else:
                st.success("No forecast hour in the next 24 hours meets this prototype's elevated screening rules.")
            st.caption(
                "The two-hour notice means the weather forecast reaches an illustrative temperature/humidity trigger; "
                "it cannot predict when an individual health incident will happen."
            )

        daily_forecast = (
            forecast.assign(date=forecast["time"].dt.strftime("%Y-%m-%d"))
            .groupby("date", as_index=False)
            .agg(
                forecast_max_temp_c=("temperature_c", "max"),
                forecast_min_temp_c=("temperature_c", "min"),
                forecast_max_humidity_pct=("relative_humidity_pct", "max"),
            )
        )
        st.markdown("**Seven-day outlook**")
        st.dataframe(daily_forecast, hide_index=True, width="stretch")
        available_dates = forecast["time"].dt.date.drop_duplicates().tolist()
        chosen_date = st.date_input(
            "Choose a forecast date",
            value=available_dates[0],
            min_value=available_dates[0],
            max_value=available_dates[-1],
            help="Open the calendar and select any date covered by the seven-day forecast.",
        )
        selected_day = forecast.loc[forecast["time"].dt.date == chosen_date].copy()
        if not selected_day.empty:
            st.markdown(f"**Hourly forecast for {chosen_date:%A, %d %B %Y} (Uganda local time)**")
            show_weather_charts(selected_day)
            radiation_fields = [field for field in ("direct_radiation", "diffuse_radiation") if field in selected_day]
            if len(radiation_fields) == 2:
                radiation_totals = selected_day[radiation_fields].clip(lower=0).sum()
                if radiation_totals.sum() > 0:
                    radiation_pie_data = pd.DataFrame(
                        {"Component": ["Direct horizontal", "Diffuse"], "Forecast radiation (Wh/m²)": radiation_totals.values}
                    )
                    radiation_pie = px.pie(
                        radiation_pie_data,
                        names="Component",
                        values="Forecast radiation (Wh/m²)",
                        title="Daily forecast radiation mix (estimate)",
                    )
                    st.plotly_chart(radiation_pie, width="stretch")
            available_times = selected_day["time"].tolist()
            if not is_public_view:
                chosen_time = st.selectbox(
                    "Choose forecast hour for map",
                    available_times,
                    format_func=lambda value: value.strftime("%Y-%m-%d %H:%M EAT"),
                )
                map_spacing = st.select_slider("Map sample spacing (illustrative)", options=[5, 10, 25], value=10, format_func=lambda value: f"{value} km")
                if st.button("Build approximate forecast map"):
                    try:
                        map_grid = fetch_forecast_grid(latitude, longitude, map_spacing, chosen_time)
                        st.map(map_grid, latitude="lat", longitude="lon", color="risk_color", zoom=9)
                        st.dataframe(map_grid, hide_index=True, width="stretch")
                    except (requests.RequestException, ValueError, KeyError) as error:
                        st.error(f"Could not build the approximate forecast map: {error}")
                st.caption(
                    "The forecast map samples a few nearby model grid points. It is not satellite imagery, "
                    "a high-resolution risk surface, or a validated GIS prediction; nearby points may share the same model grid cell."
                )
            selected_day["time"] = selected_day["time"].dt.strftime("%H:%M")
            st.dataframe(selected_day, hide_index=True, width="stretch")
        st.caption("Forecasts are cached for up to 15 minutes. Use “Update forecast now” in the sidebar to request a fresh result sooner.")
    elif data_mode == "Past hourly weather (Open-Meteo archive)":
        historical_range = st.date_input(
            "Past date range (archive availability may lag)",
            value=(datetime.now().date() - timedelta(days=14), datetime.now().date() - timedelta(days=14)),
            max_value=datetime.now().date() - timedelta(days=6),
        )
        if not isinstance(historical_range, tuple) or len(historical_range) != 2:
            st.info("Select both a start date and an end date in the calendar.")
            st.stop()
        historical_start, historical_end = historical_range
        if (historical_end - historical_start).days > 31:
            st.error("Choose a historical range of 31 days or less to keep requests responsive.")
            st.stop()
        try:
            historical_data = fetch_open_meteo_archive(
                latitude, longitude, historical_start, historical_end
            )
        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(f"Could not retrieve archived weather: {error}")
            st.stop()
        latest = historical_data.iloc[-1]
        temp = float(latest["temperature_c"])
        humidity = float(latest["relative_humidity_pct"])
        observation_time = latest["time"].to_pydatetime()
        source_label = "Open-Meteo historical reanalysis estimate"
        st.warning("Historical archive values are reanalysis/model estimates, not direct observations from a station or satellite images.")
        show_weather_charts(historical_data)
        st.download_button(
            "Download historical hourly data as CSV",
            historical_data.to_csv(index=False),
            "agro_sentinel_historical_weather.csv",
            "text/csv",
        )
        if {"direct_radiation", "diffuse_radiation"}.issubset(historical_data.columns):
            radiation_totals = historical_data[["direct_radiation", "diffuse_radiation"]].clip(lower=0).sum()
            if radiation_totals.sum() > 0:
                pie_data = pd.DataFrame(
                    {"Component": ["Direct horizontal", "Diffuse"], "Historical estimate (Wh/m²)": radiation_totals.values}
                )
                st.plotly_chart(
                    px.pie(pie_data, names="Component", values="Historical estimate (Wh/m²)", title="Daily radiation mix (reanalysis estimate)"),
                    width="stretch",
                )
        st.dataframe(historical_data, hide_index=True, width="stretch")
    else:
        uploaded_file = st.file_uploader("Upload an official station CSV", type=["csv"])
        st.caption(
            "CSV must contain temperature_c and relative_humidity_pct columns. "
            "Optional columns: timestamp and station."
        )
        if uploaded_file is None:
            st.info("Choose a station CSV file to display its latest row.")
            st.stop()
        try:
            station_data = pd.read_csv(uploaded_file)
            required_columns = {"temperature_c", "relative_humidity_pct"}
            if not required_columns.issubset(station_data.columns):
                raise ValueError("CSV needs temperature_c and relative_humidity_pct columns.")
            station_data = station_data.dropna(subset=list(required_columns))
            if station_data.empty:
                raise ValueError("CSV contains no rows with both temperature and humidity.")
            if "timestamp" in station_data.columns:
                station_data["timestamp"] = pd.to_datetime(station_data["timestamp"], errors="raise")
                station_data = station_data.sort_values("timestamp")
            latest = station_data.iloc[-1]
            temp = float(latest["temperature_c"])
            humidity = float(latest["relative_humidity_pct"])
            if not -10 <= temp <= 60 or not 0 <= humidity <= 100:
                raise ValueError("Latest row values are outside expected temperature/humidity ranges.")
            if "timestamp" in station_data.columns and pd.notna(latest.get("timestamp")):
                observation_time = latest["timestamp"].to_pydatetime()
            if "station" in station_data.columns and pd.notna(latest.get("station")):
                station_name = str(latest["station"])
            source_label = "uploaded station CSV (source not independently verified)"
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError, TypeError) as error:
            st.error(f"Could not read station CSV: {error}")
            st.stop()

# Transparent, illustrative screening rules; thresholds must be reviewed against applicable guidance.
if temp >= 38 or (temp >= 35 and humidity >= 70):
    risk = "HIGH"
    explanation = "High air temperature, or elevated temperature combined with high humidity."
    action = "Pause strenuous outdoor work, move to a shaded/cool area, hydrate, and have a supervisor reassess conditions."
    work_rest = "Use a conservative work pause and reassess frequently; do not treat a fixed interval as medically validated."
elif temp >= 32 or (temp >= 29 and humidity >= 75):
    risk = "ELEVATED"
    explanation = "Warm conditions or warm conditions combined with high humidity increase heat strain concerns."
    action = "Increase shade and water breaks, reduce heavy tasks during the hottest period, and monitor workers."
    work_rest = "Schedule more frequent shaded rest breaks and reassess after changes in conditions or workload."
else:
    risk = "LOWER SCREENING LEVEL"
    explanation = "The entered air temperature and humidity are below this prototype's elevated screening triggers."
    action = "No example trigger was crossed; this does not establish that conditions are safe. Continue local monitoring and follow workplace guidance."
    work_rest = "Continue planned work with routine breaks and ongoing monitoring."

# Workload modifies the cautionary message, not a validated physiological model.
if workload == "Heavy":
    workload_note = "Heavy work increases heat strain; apply extra caution and consider reducing intensity."
elif workload == "Moderate":
    workload_note = "Moderate work still requires water, shade, and monitoring."
else:
    workload_note = "Light work does not eliminate heat risk; maintain water and shade access."

is_forecast = source_label.startswith("Open-Meteo")
if is_public_view:
    st.subheader(f"Weather overview for {station_name}")
else:
    st.subheader("2. Explainable forecast screening" if is_forecast else "2. Explainable heat-risk screening")
a, b, c = st.columns(3)
a.metric("Forecast temperature" if is_forecast else "Air temperature estimate", f"{temp:.1f} °C")
b.metric("Forecast humidity" if is_forecast else "Relative humidity estimate", f"{humidity:.0f}%")
if is_public_view:
    c.metric("Weather data", "Forecast" if is_forecast else "Past estimate")
    st.info(
        "Use weather information for planning. It cannot predict illness or determine whether conditions are safe for a person. "
        "For work decisions, use an appropriate on-site instrument and your supervisor's approved local procedures."
    )
else:
    c.metric("Example category (not validated)", risk)
    st.write("**Why this example category was assigned:**", explanation)
    st.write("**Workload consideration:**", workload_note)
    st.write("**Example response prompt (not validated guidance):**", action)

    st.subheader("3. Example break-planning prompt")
    st.write(work_rest)
    st.caption(
        "This is not a certified work-rest schedule. Actual schedules should follow applicable occupational safety guidance and account "
        "for workload, clothing, acclimatisation, worker condition, shade and available water."
    )

if not is_public_view:
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "observation_time": observation_time.isoformat(timespec="seconds"),
        "station": station_name,
        "temperature_c": temp,
        "relative_humidity_pct": humidity,
        "workload": workload,
        "screening_category": risk,
        "data_source": source_label,
    }
    if st.button("Save this data point to local CSV", type="primary"):
        df_new = pd.DataFrame([record])
        if DATA_FILE.exists():
            df_new.to_csv(DATA_FILE, mode="a", header=False, index=False)
        else:
            df_new.to_csv(DATA_FILE, index=False)
        st.success(f"Saved data record to {DATA_FILE.name}")

    st.subheader("4. Local data log")
    if DATA_FILE.exists():
        history = pd.read_csv(DATA_FILE)
        st.dataframe(history.tail(20), width="stretch")
        st.download_button("Download CSV log", history.to_csv(index=False), "agro_sentinel_readings.csv", "text/csv")
    else:
        st.info("No readings saved yet. Save a reading to create the local CSV log.")

st.subheader("5. Location map")
map_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})

if is_public_view:
    public_map = folium.Map(location=[latitude, longitude], zoom_start=11)
    folium.Marker([latitude, longitude], tooltip="Selected location").add_to(public_map)
    map_selection = st_folium(public_map, height=420, width=None, key="public_location_map")
    if map_selection and map_selection.get("last_clicked"):
        clicked = map_selection["last_clicked"]
        clicked_lat = float(clicked["lat"])
        clicked_lon = float(clicked["lng"])
        if abs(clicked_lat - latitude) > 0.00001 or abs(clicked_lon - longitude) > 0.00001:
            st.session_state.site_lat = clicked_lat
            st.session_state.site_lon = clicked_lon
            st.rerun()
    st.caption("Tap or click a location on the map to update both latitude and longitude.")
else:
    st.map(map_df, latitude="lat", longitude="lon", zoom=11)
    st.caption("Operator map marker. It shows the entered point, not a verified field sensor location.")

st.write(f"Selected location: {latitude:.5f}° latitude, {longitude:.5f}° longitude")

if not is_public_view:
    st.subheader("6. System status and next integration steps")
    st.write(
        "- Current status: laptop-based software demonstrator; no ESP32 or physical sensors connected.\n"
        "- Data shown: manual input, generated samples, NASA POWER historical/near-real-time estimates, Open-Meteo forecasts and archive, or uploaded station CSV rows.\n"
        "- Next hardware integration: sensor → ESP32 → timestamped CSV/serial data → dashboard.\n"
        "- Uganda Meteorological Authority: no direct connection configured; use the CSV upload or obtain an approved API/data format.\n"
        "- GIS forecast surface: optional coarse nine-point model estimate around the selected coordinates; not high-resolution satellite imagery or a validated risk map.\n"
        "- Predictive capability: weather forecasts and transparent screening rules are planning aids, not predictions of individual incidents or validated safety classifications.\n"
        "- WBGT: not calculated. Air temperature and relative humidity alone are insufficient for a "
        "standards-compliant WBGT measurement."
    )
    with st.expander("What is needed before claiming incident predictions?"):
        st.markdown(
            "1. **Define one measurable outcome** with local health and worker-safety experts—for example, an anonymized, "
            "site-level count of heat-related events per shift, not a prediction about a named individual.\n"
            "2. **Collect permitted local evidence:** calibrated on-site WBGT and weather observations, shift/workload context, "
            "and reliably recorded outcome timestamps. Use consent, ethics review, data minimization, access controls and retention rules.\n"
            "3. **Build and compare a model** against simple baselines using earlier periods for training and later periods/sites "
            "for independent testing. Check calibration, missed-event rate, false alarms, subgroup performance and seasonal drift.\n"
            "4. **Run a prospective pilot with human review** before any operational use. Document uncertainty and keep emergency "
            "and occupational decisions with qualified people.\n\n"
            "Weather forecasts alone can forecast weather conditions, not individual illness or a future incident. This app currently "
            "does not train or validate an incident-prediction model."
        )
else:
    st.caption(
        "This page provides weather information and planning context only. A below-trigger result is not a safety clearance; "
        "a forecast cannot tell whether an individual will have a health incident."
    )
st.caption(
    "Data attribution: forecasts and historical reanalysis from Open-Meteo (CC BY 4.0; "
    "see open-meteo.com/en/licence). NASA POWER meteorology and satellite-derived "
    "surface-radiation estimates from NASA POWER (see power.larc.nasa.gov)."
)
