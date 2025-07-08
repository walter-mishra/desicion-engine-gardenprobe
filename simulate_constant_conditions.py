import argparse
import numpy as np
import pandas as pd

from greenlight_gym.envs.greenlight import GreenLightStatesTest
from greenlight_gym.common.utils import rh2vaporDens, vaporDens2pres, co2ppm2dens


def generate_weather(temp, rh, co2_ppm, light, season_length, time_interval):
    """Generate constant weather data array."""
    c = 86400
    n_steps = int(season_length * c / time_interval)
    # add one step for final observation
    weather = np.zeros((n_steps + 1, 10), dtype=np.float32)
    weather[:, 0] = light  # global radiation W/m^2
    weather[:, 1] = temp   # outdoor temperature deg C
    vp_density = rh2vaporDens(temp, rh)
    weather[:, 2] = vaporDens2pres(temp, vp_density)
    weather[:, 3] = co2ppm2dens(temp, co2_ppm) * 1e6
    weather[:, 4] = 0.0    # wind speed
    weather[:, 5] = temp - 5.0  # sky temperature
    weather[:, 6] = temp        # soil temperature
    weather[:, 7] = light * time_interval / 1e6  # daily radiation sum approx
    weather[:, 8] = 1.0  # isDay
    weather[:, 9] = 1.0  # isDaySmooth
    return weather


def run_simulation(args):
    weather = generate_weather(
        args.temperature,
        args.humidity,
        args.co2,
        args.light,
        args.season_length,
        args.time_interval,
    )

    env = GreenLightStatesTest(
        weather_data_dir="",
        location="",
        data_source="",
        nx=28,
        nu=8,
        nd=10,
        no_lamps=0,
        led_lamps=1,
        hps_lamps=0,
        int_lamps=0,
        h=args.h,
        season_length=args.season_length,
        pred_horizon=0,
        time_interval=args.time_interval,
        training=False,
        start_train_year=2000,
        end_train_year=2000,
        start_train_day=1,
        end_train_day=1,
        control_signals=[
            "uBoil",
            "uCO2",
            "uThScr",
            "uVent",
            "uLamp",
            "uIntLamp",
            "uGroPipe",
            "uBlScr",
        ],
        model_obs_vars=[
            "co2Air",
            "co2Top",
            "tAir",
            "tTop",
            "tCan",
            "tCovIn",
            "tCovE",
            "tThScr",
            "tFlr",
            "tPipe",
            "tSo1",
            "tSo2",
            "tSo3",
            "tSo4",
            "tSo5",
            "vpAir",
            "vpTop",
            "tLamp",
            "tIntLamp",
            "tGroPipe",
            "tBlScr",
            "tCan24",
            "cBuf",
            "cLeaf",
            "cStem",
            "cFruit",
            "tCanSum",
            "Time",
        ],
        weather_obs_vars=[],
        weather=weather,
    )

    obs, _ = env.reset()
    n_steps = env.N
    data = np.zeros((n_steps + 1, len(obs)))
    data[0] = obs

    for i in range(n_steps):
        obs, _, terminated, _, _ = env.step(np.zeros(env.action_space.shape))
        data[i + 1] = obs
        if terminated:
            break

    df = pd.DataFrame(data, columns=[
        "co2Air","co2Top","tAir","tTop","tCan","tCovIn","tCovE","tThScr","tFlr",
        "tPipe","tSo1","tSo2","tSo3","tSo4","tSo5","vpAir","vpTop","tLamp",
        "tIntLamp","tGroPipe","tBlScr","tCan24","cBuf","cLeaf","cStem",
        "cFruit","tCanSum","Time"])

    print(df[["cLeaf","cStem","cFruit","Time"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GreenLight simulation with constant climate")
    parser.add_argument("--temperature", type=float, default=20.0, help="Outdoor temperature [C]")
    parser.add_argument("--humidity", type=float, default=70.0, help="Outdoor relative humidity [%]")
    parser.add_argument("--co2", type=float, default=400.0, help="Outdoor CO2 concentration [ppm]")
    parser.add_argument("--light", type=float, default=200.0, help="Global radiation [W m^-2]")
    parser.add_argument("--season_length", type=int, default=1, help="Season length [days]")
    parser.add_argument("--time_interval", type=int, default=300, help="Time interval [s]")
    parser.add_argument("--h", type=float, default=1.0, help="Solver step size [s]")
    args = parser.parse_args()

    run_simulation(args)
