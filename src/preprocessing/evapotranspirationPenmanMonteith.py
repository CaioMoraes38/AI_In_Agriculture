import pandas as pd
import numpy as np
from datetime import datetime
import math
import os

SOROCABA_LAT = -23.5018
SOROCABA_LON = -47.4711
SOROCABA_ELEVATION = 580

kc_dict = {
    'Wheat': {'Initial': 0.30, 'Vegetative': 0.75, 'Mid': 1.15, 'Late': 0.30},
    'Maize': {'Initial': 0.30, 'Vegetative': 0.80, 'Mid': 1.20, 'Late': 0.45},
    'Cotton': {'Initial': 0.35, 'Vegetative': 0.75, 'Mid': 1.18, 'Late': 0.60},
    'Rice': {'Initial': 1.05, 'Vegetative': 1.10, 'Mid': 1.20, 'Late': 0.75},
    'Sugarcane': {'Initial': 0.40, 'Vegetative': 0.80, 'Mid': 1.25, 'Late': 0.75},
    'Potato': {'Initial': 0.50, 'Vegetative': 0.80, 'Mid': 1.15, 'Late': 0.75}
}


def get_kc_value(crop_type, growth_stage):
    crop_type = crop_type.strip()
    growth_stage = growth_stage.strip()
    
    if crop_type in kc_dict:
        if growth_stage in kc_dict[crop_type]:
            return kc_dict[crop_type][growth_stage]
        else:
            return np.mean(list(kc_dict[crop_type].values()))
    else:
        return 0.5


def calculate_atmospheric_pressure(elevation):
    return 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26


def calculate_saturation_vapor_pressure(temperature):
    return 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))


def calculate_actual_vapor_pressure(temperature, humidity):
    es = calculate_saturation_vapor_pressure(temperature)
    return (humidity / 100.0) * es


def calculate_extraterrestrial_radiation(latitude, day_of_year):
    Gsc = 0.0820
    
    b = 2 * math.pi * (day_of_year - 1) / 365.0
    dr = 1.0 + 0.033 * math.cos(b)
    
    delta = 0.409 * math.sin(b - 1.39)
    
    phi = math.radians(latitude)
    
    ws = math.acos(-math.tan(phi) * math.tan(delta))
    
    Ra = (24 * 60 / math.pi) * Gsc * dr * (ws * math.sin(phi) * math.sin(delta) + 
                                            math.cos(phi) * math.cos(delta) * math.sin(ws))
    
    return Ra


def calculate_solar_radiation(sunlight_hours, day_of_year, latitude):
    Ra = calculate_extraterrestrial_radiation(latitude, day_of_year)
    
    b = 2 * math.pi * (day_of_year - 1) / 365.0
    delta = 0.409 * math.sin(b - 1.39)
    phi = math.radians(latitude)
    ws = math.acos(-math.tan(phi) * math.tan(delta))
    N = (24 / math.pi) * ws
    
    as_coeff = 0.25
    bs_coeff = 0.50
    
    Rs = (as_coeff + bs_coeff * (sunlight_hours / N)) * Ra
    
    return Rs


def calculate_net_radiation(solar_radiation, temperature):
    alpha = 0.23
    
    Rns = (1 - alpha) * solar_radiation
    
    Rnl = 2.042e-10 * ((temperature + 273.16) ** 4)
    
    return Rns, Rnl


def penman_monteith(temperature, humidity, wind_speed, sunlight_hours, 
                    day_of_year=None, latitude=SOROCABA_LAT, elevation=SOROCABA_ELEVATION):
    Cn = 900
    Cd = 0.34
    
    if day_of_year is None:
        day_of_year = datetime.now().timetuple().tm_yday
    
    u2 = wind_speed / 3.6
    
    P = calculate_atmospheric_pressure(elevation)
    
    lambda_val = 2.45
    gamma = 0.665 * P / 1000.0
    
    es = calculate_saturation_vapor_pressure(temperature)
    ea = calculate_actual_vapor_pressure(temperature, humidity)
    
    vpd = es - ea
    
    Rs = calculate_solar_radiation(sunlight_hours, day_of_year, latitude)
    
    Rns, Rnl = calculate_net_radiation(Rs, temperature)
    Rn = Rns - Rnl
    
    Rn_mm = Rn / lambda_val * 1000
    
    delta = 4098 * es / ((temperature + 237.3) ** 2)
    
    numerator = 0.408 * delta * Rn_mm + gamma * (Cn / (temperature + 273)) * u2 * vpd
    denominator = delta + gamma * (1 + Cd * u2)
    
    ETo = numerator / denominator
    
    return max(ETo, 0)


def process_irrigation_data(input_file, output_file):
    try:
        df = pd.read_csv(input_file, skipinitialspace=True, on_bad_lines='warn')
    except:
        df = pd.read_csv(input_file, skipinitialspace=True, on_bad_lines='skip')
    
    df.columns = df.columns.str.strip()
    
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].str.strip()
    
    df['Kc'] = df.apply(lambda row: get_kc_value(row['Crop_Type'], row['Crop_Growth_Stage']), axis=1)
    
    df['ETo_mm_dia'] = df.apply(
        lambda row: penman_monteith(
            temperature=row['Temperature_C'],
            humidity=row['Humidity'],
            wind_speed=row['Wind_Speed_kmh'],
            sunlight_hours=row['Sunlight_Hours']
        ),
        axis=1
    )
    
    df['ETc_mm_dia'] = df['ETo_mm_dia'] * df['Kc']
    
    df.to_csv(output_file, index=False)
    
    print(f"Arquivo processado com sucesso!")
    print(f"Entrada: {input_file}")
    print(f"Saída: {output_file}")
    print(f"\nColunas adicionadas:")
    print(f"  - Kc (Coeficiente de Cultura)")
    print(f"  - ETo_mm_dia (Evapotranspiração de Referência em mm/dia)")
    print(f"  - ETc_mm_dia (Evapotranspiração do Cultivo em mm/dia)")
    print(f"\nCoordenadas utilizadas (Sorocaba, SP):")
    print(f"  - Latitude: {SOROCABA_LAT}°")
    print(f"  - Longitude: {SOROCABA_LON}°")
    print(f"  - Elevação: {SOROCABA_ELEVATION} m")
    
    print(f"\nEstatísticas dos resultados:")
    print(f"\nKc:")
    print(df['Kc'].describe())
    print(f"\nETo_mm_dia:")
    print(df['ETo_mm_dia'].describe())
    print(f"\nETc_mm_dia:")
    print(df['ETc_mm_dia'].describe())
    
    return df


if __name__ == "__main__":
    # Caminhos dos arquivos na mesma pasta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "irrigation_prediction.csv")
    output_path = os.path.join(current_dir, "irrigation_prediction_with_etc.csv")
    
    # Processar dados
    result_df = process_irrigation_data(input_path, output_path)
    
    # Exibir primeiras linhas
    print("\nPrimeiras linhas do arquivo processado:")
    print(result_df[['Crop_Type', 'Crop_Growth_Stage', 'Temperature_C', 'Humidity', 
                      'Wind_Speed_kmh', 'Sunlight_Hours', 'Kc', 'ETo_mm_dia', 'ETc_mm_dia']].head())
