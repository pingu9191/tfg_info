from enum import Enum

raw_channels = ["SessionTime",
"SessionTick",
"SessionNum",
"SessionState",
"SessionFlags",
"DriverMarker",
"PushToPass",
"IsReplayPlaying",
"PlayerTrackSurface",
"PlayerCarTeamIncidentCount",
"PlayerCarWeightPenalty",
"PlayerCarPowerAdjust",
"PlayerTireCompound",
"PaceMode",
"SteeringWheelAngle",
"Throttle",
"Brake",
"Clutch",
"Gear",
"RPM",
"Lap",
"LapDistPct",
"LapLastLapTime",
"Speed",
"Yaw",
"YawNorth",
"Pitch",
"Roll",
"TrackTempCrew",
"AirTemp",
"TrackWetness",
"Skies",
"AirDensity",
"AirPressure",
"WindVel",
"WindDir",
"RelativeHumidity",
"FogLevel",
"Precipitation",
"DCLapStatus",
"DCDriversSoFar",
"ThrottleRaw",
"BrakeRaw",
"ClutchRaw",
"HandbrakeRaw",
"BrakeABSactive",
"RollRate_ST[0]",
"RollRate_ST[1]",
"RollRate_ST[2]",
"RollRate_ST[3]",
"RollRate_ST[4]",
"RollRate_ST[5]",
"YawRate",
"PitchRate",
"RollRate",
"VertAccel",
"LatAccel",
"LongAccel",
"dcStarter",
"dcPitSpeedLimiterToggle",
"dcTearOffVisor",
"dcBrakeBias",
"RFbrakeLinePress",
"RFcoldPressure",
"RFtempCL",
"RFtempCM",
"RFtempCR",
"RFwearL",
"RFwearM",
"RFwearR",
"LFbrakeLinePress",
"LFcoldPressure",
"LFtempCL",
"LFtempCM",
"LFtempCR",
"LFwearL",
"LFwearM",
"LFwearR",
"FuelLevel",
"RRbrakeLinePress",
"RRcoldPressure",
"RRtempCL",
"RRtempCM",
"RRtempCR",
"RRwearL",
"RRwearM",
"RRwearR",
"LRbrakeLinePress",
"LRcoldPressure",
"LRtempCL",
"LRtempCM",
"LRtempCR",
"LRwearL",
"LRwearM",
"LRwearR",
"LRshockDefl",
"LRshockVel",
"RRshockDefl",
"RRshockVel",
"LFshockDefl",
"LFshockVel",
"RFshockDefl",
"RFshockVel"]

model_channels = [#"SessionTime",
#"SessionTick",
#"SessionNum",
#"SessionState",
#"SessionFlags",
#"DriverMarker",
"PushToPass",
#"IsReplayPlaying",
"PlayerTrackSurface",
#"PlayerCarTeamIncidentCount",
#"PlayerCarWeightPenalty",
"PlayerCarPowerAdjust",
"PlayerTireCompound",
"PaceMode",
"SteeringWheelAngle",
"Throttle",
"Brake",
"Clutch",
"Gear",
"RPM",
#"Lap",
"LapDistPct",
#"LapLastLapTime",
"Speed",
"Yaw",
"YawNorth",
"Pitch",
"Roll",
"TrackTempCrew",
"AirTemp",
"TrackWetness",
"Skies",
"AirDensity",
"AirPressure",
"WindVel",
"WindDir",
"RelativeHumidity",
"FogLevel",
"Precipitation",
#"DCLapStatus",
#"DCDriversSoFar",
"ThrottleRaw",
"BrakeRaw",
"ClutchRaw",
"HandbrakeRaw",
"BrakeABSactive",
"RollRate_ST[0]",
"RollRate_ST[1]",
"RollRate_ST[2]",
"RollRate_ST[3]",
"RollRate_ST[4]",
"RollRate_ST[5]",
"YawRate",
"PitchRate",
"RollRate",
"VertAccel",
"LatAccel",
"LongAccel",
#"dcStarter",
#"dcPitSpeedLimiterToggle",
#"dcTearOffVisor",
"dcBrakeBias",
"RFbrakeLinePress",
"RFcoldPressure",
"RFtempCL",
"RFtempCM",
"RFtempCR",
"RFwearL",
"RFwearM",
"RFwearR",
"LFbrakeLinePress",
"LFcoldPressure",
"LFtempCL",
"LFtempCM",
"LFtempCR",
"LFwearL",
"LFwearM",
"LFwearR",
"FuelLevel",
"RRbrakeLinePress",
"RRcoldPressure",
"RRtempCL",
"RRtempCM",
"RRtempCR",
"RRwearL",
"RRwearM",
"RRwearR",
"LRbrakeLinePress",
"LRcoldPressure",
"LRtempCL",
"LRtempCM",
"LRtempCR",
"LRwearL",
"LRwearM",
"LRwearR",
"LRshockDefl",
"LRshockVel",
"RRshockDefl",
"RRshockVel",
"LFshockDefl",
"LFshockVel",
"RFshockDefl",
"RFshockVel"]

class LapType(Enum):
    """
    Enum for lap types.
    
    - VALID_LAP: A valid lap.
    - INCOMPLETE_LAP: An incomplete lap.
    - INLAP_LAP: An in-lap (starting out of the pitline).
    - OUTLAP_LAP: An out-lap (pitting in).
    - OFFTRACK_LAP: A lap, with at least, one off-track.
    """
    VALID_LAP = 0
    INCOMPLETE_LAP = 1
    INLAP_LAP = 2
    OUTLAP_LAP = 3
    OFFTRACK_LAP = 4
    INCIDENT_LAP = 5

    # String representation of the enum
    def __str__(self):
        return self.name
    
class Track():
    """
    Class for track.
    
    - name: Name of the track.
    - length: Length of the track in meters.
    - number_of_sections: Number of sections in the track.
    - sections: List of track sections.
    """
    def __init__(self, name, length, sections):
        self.name = name
        self.length = length
        self.number_of_sections = len(sections)
        self.sections = []
        for section in sections:
            self.sections.append(TrackSection(section["sector"], 
                                              section["start"]/length, 
                                              section["end"]/length))
    
class TrackSection():
    """
    Class for track sections.
    
    - name: Name of the section.
    - length: Length of the section in meters.
    - start: Start of the section in meters.
    - end: End of the section in meters.
    """
    def __init__(self, name, start, end):
        self.name = str(name)
        self.length = end - start
        self.start = start
        self.end = end
    