# Configuration data for an imaging sonar system which uses an array of
# multiple acoustic transducers to emit sound waves and measure the
# reflection of sound waves in order to generate an image.
# This message is not appropriate for simpler sonar systems like
# depth sounders or side-scan sonars.
# This message is used by the sonar to report the current state of the
# system.
# Source: Bastian Daniel Schildknecht (https://github.com/bastianschildknecht/sonar_driver_interfaces)

# Fire mode constants
uint8 FIRE_MODE_LOW_FREQUENCY = 1
uint8 FIRE_MODE_HIGH_FREQUENCY = 2

# Gamma correction constants
uint8 GAMMA_CORRECTION_NONE = 255
uint8 GAMMA_CORRECTION_LOW = 192
uint8 GAMMA_CORRECTION_MEDIUM = 127
uint8 GAMMA_CORRECTION_HIGH = 63

# Ping rate constants
uint8 PING_RATE_STANDBY = 0
uint8 PING_RATE_VERY_LOW = 2
uint8 PING_RATE_LOW = 5
uint8 PING_RATE_MEDIUM = 10
uint8 PING_RATE_HIGH = 15
uint8 PING_RATE_VERY_HIGH = 40

# Network speed limit constants
uint8 NET_SPEED_UNLIMITED = 255
uint8 NET_SPEED_VERY_HIGH = 192
uint8 NET_SPEED_HIGH = 127
uint8 NET_SPEED_MEDIUM = 63
uint8 NET_SPEED_LOW = 31
uint8 NET_SPEED_VERY_LOW = 10
uint8 NET_SPEED_LOWEST = 5



std_msgs/Header header      # Timestamp in the header is the time of message
                            # message transmission.

uint8 fire_mode             # The mode of the sonar system.
                            # 1: Low frequency
                            # 2: High frequency

float64 frequency           # The frequency of the sound waves emitted by the
                            # sonar system. [Hz]

uint8 ping_rate             # The rate at which the sonar system sends out
                            # pings [Hz].

uint16 beam_count           # The number of beams in the sonar system.
float64 beam_separation     # The seperation between adjacent beams [degrees].

float64 min_range           # The minimum range [m].
float64 max_range           # The maximum range [m].
float64 current_range       # The currently set range [m].
float64 range_resolution    # The range resolution [m].
uint32 range_count          # The number of range bins.

float64 horz_fov            # The horizontal field of view [degrees].
float64 vert_fov            # The vertical field of view [degrees].
float64 angular_resolution  # The angular resolution [degrees].

float64 gain                # The gain of the sonar system [%].
bool gain_assist            # Whether the gain is assisted or not.

uint8 gamma                 # The gamma correction of the sonar system.
                            # 0 and 255 are gamma 1.0.
                            # 127 is gamma 0.5.

float64 speed_of_sound      # The speed of sound [m/s]
float64 salinity            # The salinity of the water [ppt]
float64 temperature         # The temperature of the water [degC]
float64 pressure            # The pressure of the water [bar]

uint8 net_speed_limit       # The network speed limit [Mbps].
                            # 0 or 255 means no limit.