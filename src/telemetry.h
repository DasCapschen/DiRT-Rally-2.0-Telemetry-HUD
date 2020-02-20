#include <string.h> //memset

struct Vec3 {
    float x;
    float y;
    float z;
};

struct Wheels {
    float bl;
    float br;
    float fl;
    float fr;
};

struct TelemetryStruct {
    float time;                   // 0 since race started???
    float lap_time;               // 4 for rally -> total time
    float lap_distance;           // 8 for rally -> total distance in meters
    float total_distance;         // 12 in meters
    Vec3 world_position;          // 16
    float speed;                  // 28 in meters per second
    Vec3 velocity;                // 32 
    Vec3 roll;                    // 44 right vector, or maybe left(why?) vector
    Vec3 pitch;                   // 56 forward vector
    Wheels suspension_pos;        // 68
    Wheels suspension_vel;        // 84
    Wheels wheel_speed;           // 100
    float throttle;               // 116 0 to 1, 1 = full throttle
    float steering;               // 120 -1 = full left, 1 = full right ; 540°
    float brake;                  // 124 0 to 1, 1 = hard braking
    float clutch;                 // 128 0 to 1, 1 = driver uses clutch
    float gear;                   // 132 -1 is reverse, else 1 to max_gears
    float gforce_lat;             // 136
    float gforce_lon;             // 140
    float current_lap;            // 144 starts at 0
    float rpm;                    // 148 rpm divided by 10
    float sli_support;            // 152 always one / unused?
    float race_position;          // 156 starts at 1
    float kers_level;             // 160 unused
    float kers_max;               // 164 unused
    float drs;                    // 168 unused
    float tc;                     // 172 unused
    float abs;                    // 176 unused
    float fuel_in_tank;           // 180 unused
    float fuel_capacity;          // 184 unused
    float in_pit;                 // 188 unused
    float sector;                 // 192 starts at 0
    float time_sector_1;          // 196 where are sector 3 and 4???
    float time_sector_2;          // 200 these do not get overridden!
    Wheels brake_temp;            // 204 starts negative, lerps toward ambient?
    Wheels wheel_pressure;        // 220 unused, always 0
    float completed_laps;         // 236 for rally -> 1 if finished
    float total_laps;             // 240 for rally -> 1
    float track_length;           // 244 in meters
    float last_lap_time;          // 248 for rally -> total time in seconds
    float max_rpm;                // 252 max rpm divided by 10
    float idle_rpm;               // 256 idle rpm divided by 10
    float max_gears;              // 260 
}; //264 bytes

union TelemetryUnion {
    char raw[264];
    TelemetryStruct tel;
};

static TelemetryStruct telemetry_from_buffer(char buf[264]) 
{
    TelemetryUnion u = {0};
    memcpy( u.raw, buf, 264 );
    return u.tel;
}