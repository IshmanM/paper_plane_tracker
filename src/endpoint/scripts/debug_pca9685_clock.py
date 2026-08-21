import statistics
import threading
import time

import board
from adafruit_pca9685 import PCA9685
from gpiozero import DigitalInputDevice

import src.endpoint.config as config


DEFAULT_GPIO_BCM = 17
DEFAULT_PCA_CHANNEL = 15
DEFAULT_MEASUREMENT_SECONDS = 10.0

# Deliberately use the PCA9685 nominal clock here. The point of this script is
# to estimate the real oscillator clock from the measured PWM frequency.
NOMINAL_REFERENCE_CLOCK_HZ = 25_000_000


def _promptYesNo(prompt: str) -> bool:
    return input(f"{prompt} [y/N]: ").strip().lower() in ("y", "yes")


def _promptInt(prompt: str, default: int) -> int:
    text = input(f"{prompt} [{default}]: ").strip()
    return default if not text else int(text)


def _promptFloat(prompt: str, default: float) -> float:
    text = input(f"{prompt} [{default:g}]: ").strip()
    return default if not text else float(text)


def _prescaleForFrequency(reference_clock_hz: float, frequency_hz: float) -> int:
    return int(reference_clock_hz / (4096.0 * frequency_hz) + 0.5) - 1


def main() -> None:
    print("PCA9685 reference-clock measurement")
    print()
    print("SAFETY CHECK:")
    print("The PCA9685 PWM SIGNAL connected to the Raspberry Pi GPIO must be <= 3.3 V")
    print("relative to Raspberry Pi GND. 3.0-3.15 V is fine.")
    print("Do NOT connect servo V+ to a Pi GPIO.")
    if not _promptYesNo("Have you verified the signal reaching the Pi GPIO is <= 3.3 V?"):
        print("Cancelled.")
        return

    print()
    print("Run this with the normal endpoint program stopped.")
    print("Use an UNUSED PCA9685 channel; channel 15 is the default.")
    print("Connect that channel's SIGNAL pin to the chosen Pi BCM GPIO and share GND.")
    if not _promptYesNo("Is the endpoint stopped and the selected PCA channel unused by a servo/device?"):
        print("Cancelled.")
        return

    gpio_bcm = _promptInt("Pi input GPIO (BCM numbering)", DEFAULT_GPIO_BCM)
    pca_channel = _promptInt("Unused PCA9685 output channel", DEFAULT_PCA_CHANNEL)
    measurement_seconds = _promptFloat("Measurement duration seconds", DEFAULT_MEASUREMENT_SECONDS)
    target_frequency_hz = float(config.PCA9685_FREQUENCY_HZ)

    if not 0 <= gpio_bcm <= 27: raise ValueError("GPIO must be BCM 0-27")
    if not 0 <= pca_channel <= 15: raise ValueError("PCA9685 channel must be 0-15")
    if measurement_seconds <= 0.0: raise ValueError("Measurement duration must be positive")

    prescale = _prescaleForFrequency(NOMINAL_REFERENCE_CLOCK_HZ, target_frequency_hz)
    nominal_quantized_frequency_hz = NOMINAL_REFERENCE_CLOCK_HZ / (4096.0 * (prescale + 1))

    print()
    print(f"Endpoint configured PWM frequency:     {target_frequency_hz:.6f} Hz")
    print(f"Debug nominal PCA reference clock:     {NOMINAL_REFERENCE_CLOCK_HZ / 1e6:.6f} MHz")
    print(f"Expected PCA prescale:                 {prescale}")
    print(f"Nominal quantized PWM frequency:       {nominal_quantized_frequency_hz:.6f} Hz")
    print("NOTE: the endpoint config reference-clock calibration is ignored by this test.")
    print()
    input("Press Enter to begin measurement...")

    timestamps_ns = []
    timestamps_lock = threading.Lock()

    i2c = board.I2C()
    pca = PCA9685(i2c_bus=i2c, reference_clock_speed=NOMINAL_REFERENCE_CLOCK_HZ)
    gpio = DigitalInputDevice(gpio_bcm, pull_up=None, active_state=True, bounce_time=None)

    def on_rising_edge() -> None:
        with timestamps_lock:
            timestamps_ns.append(time.monotonic_ns())

    actual_prescale = None

    try:
        pca.frequency = target_frequency_hz
        actual_prescale = int(pca.prescale_reg)
        pca.channels[pca_channel].duty_cycle = 0x8000  # 50% duty for easy edge measurement.
        gpio.when_activated = on_rising_edge

        print(f"Measuring rising edges for {measurement_seconds:g} s...")
        time.sleep(measurement_seconds)

        gpio.when_activated = None
        with timestamps_lock:
            ts = list(timestamps_ns)

    finally:
        try: pca.channels[pca_channel].duty_cycle = 0
        except Exception: pass
        try: gpio.close()
        except Exception: pass
        try: pca.deinit()
        except Exception: pass

    if len(ts) < 3:
        raise RuntimeError(f"Only {len(ts)} rising edges detected. Check GPIO number, shared ground, wiring, and signal voltage.")

    periods_s = [(b - a) * 1e-9 for a, b in zip(ts[:-1], ts[1:])]
    total_time_s = (ts[-1] - ts[0]) * 1e-9
    measured_frequency_hz = (len(ts) - 1) / total_time_s
    mean_period_s = statistics.mean(periods_s)
    median_period_s = statistics.median(periods_s)
    period_std_s = statistics.stdev(periods_s) if len(periods_s) >= 2 else 0.0

    if actual_prescale is None:
        raise RuntimeError("Failed to read PCA9685 prescale register.")

    actual_nominal_frequency_hz = NOMINAL_REFERENCE_CLOCK_HZ / (4096.0 * (actual_prescale + 1))
    estimated_reference_clock_hz = measured_frequency_hz * 4096.0 * (actual_prescale + 1)
    clock_ratio = estimated_reference_clock_hz / NOMINAL_REFERENCE_CLOCK_HZ
    period_scale_vs_nominal = actual_nominal_frequency_hz / measured_frequency_hz

    print()
    print("RESULTS")
    print(f"Expected PCA prescale:                 {prescale}")
    print(f"Actual PCA prescale register:          {actual_prescale}")
    print(f"Prescale matches expectation:          {actual_prescale == prescale}")
    print(f"Rising edges captured:                 {len(ts)}")
    print(f"Mean period:                           {mean_period_s * 1000.0:.6f} ms")
    print(f"Median period:                         {median_period_s * 1000.0:.6f} ms")
    print(f"Per-cycle period std (Linux jitter):   {period_std_s * 1000.0:.6f} ms")
    print(f"Measured PWM frequency:                {measured_frequency_hz:.6f} Hz")
    print(f"Estimated PCA9685 reference clock:     {estimated_reference_clock_hz / 1e6:.6f} MHz")
    print(f"Reference-clock ratio vs 25 MHz:       {clock_ratio:.6f}")
    print(f"PWM-period scale vs nominal:           {period_scale_vs_nominal:.6f}x")
    print()
    print("Suggested config value:")
    print(f"PCA9685_REFERENCE_CLOCK_SPEED = {round(estimated_reference_clock_hz)}")


if __name__ == "__main__":
    main()