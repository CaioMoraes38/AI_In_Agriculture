from enum import Enum


class SoilTypeEnum(str, Enum):
    clay = "Clay"
    silt = "Silt"
    sandy = "Sandy"


class CropTypeEnum(str, Enum):
    wheat = "Wheat"
    rice = "Rice"
    maize = "Maize"
    cotton = "Cotton"


class CropGrowthStageEnum(str, Enum):
    vegetative = "Vegetative"
    flowering = "Flowering"
    harvest = "Harvest"
    sowing = "Sowing"


class SeasonEnum(str, Enum):
    rabi = "Rabi"
    kharif = "Kharif"
    zaid = "Zaid"


class IrrigationTypeEnum(str, Enum):
    rainfed = "Rainfed"
    canal = "Canal"
    drip = "Drip"
    sprinkler = "Sprinkler"


class WaterSourceEnum(str, Enum):
    reservoir = "Reservoir"
    groundwater = "Groundwater"
    river = "River"
    rainwater = "Rainwater"


class MulchingEnum(str, Enum):
    yes = "Yes"
    no = "No"
