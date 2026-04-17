from pathlib import Path
from ia.services.irrigation_ml_regression_service import IrrigationMLRegressionService
from ia.services.plant_disease_service import PlantDiseaseService


class Dependencias:
    _irrigation_ml_service: IrrigationMLRegressionService = None
    _plant_disease_service: PlantDiseaseService = None
    
    @classmethod
    def obter_irrigation_ml_service(cls) -> IrrigationMLRegressionService:
        if cls._irrigation_ml_service is None:
            modelos_dir = Path(__file__).parent.parent / 'models'
            cls._irrigation_ml_service = IrrigationMLRegressionService(modelos_dir)
        return cls._irrigation_ml_service
    
    @classmethod
    def obter_plant_disease_service(cls) -> PlantDiseaseService:
        if cls._plant_disease_service is None:
            modelos_dir = Path(__file__).parent.parent / 'models'
            cls._plant_disease_service = PlantDiseaseService(modelos_dir)
        return cls._plant_disease_service
