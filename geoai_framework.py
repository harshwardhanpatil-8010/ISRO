import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import geopandas as gpd
import rasterio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    FLOOD_RISK = "flood_risk_assessment"
    SITE_SUITABILITY = "site_suitability_analysis"
    LAND_COVER = "land_cover_monitoring"
    BUFFER_ANALYSIS = "buffer_analysis"
    OVERLAY_ANALYSIS = "overlay_analysis"

@dataclass
class GeoprocessingStep:
    """Represents a single step in a geoprocessing workflow"""
    step_id: str
    operation: str
    parameters: Dict[str, Any]
    input_data: List[str]
    output_data: str
    reasoning: str
    dependencies: List[str] = None

@dataclass
class WorkflowResult:
    """Contains the results of a completed workflow"""
    workflow_id: str
    steps: List[GeoprocessingStep]
    outputs: Dict[str, Any]
    reasoning_chain: List[str]
    execution_time: float
    success: bool
    error_messages: List[str] = None

class GeoAIReasoningEngine:
    """
    Chain-of-Thought reasoning engine for geospatial analysis
    """
    
    def __init__(self, model_name: str = "mistral-7b-instruct"):
        self.model_name = model_name
        self.knowledge_base = self._load_geoprocessing_knowledge()
        self.reasoning_history = []
        
    def _load_geoprocessing_knowledge(self) -> Dict[str, Any]:
        """Load geoprocessing API documentation and workflows"""
        # This would typically load from a RAG database
        return {
            "operations": {
                "buffer": {
                    "description": "Creates buffer zones around geometric objects",
                    "parameters": ["distance", "units", "segments"],
                    "input_types": ["vector"],
                    "output_type": "vector"
                },
                "overlay": {
                    "description": "Performs spatial overlay operations",
                    "parameters": ["method", "how"],
                    "input_types": ["vector", "vector"],
                    "output_type": "vector"
                },
                "clip": {
                    "description": "Clips input features to boundary",
                    "parameters": ["boundary"],
                    "input_types": ["vector", "raster"],
                    "output_type": "vector/raster"
                },
                "slope_analysis": {
                    "description": "Calculates slope from elevation data",
                    "parameters": ["units", "z_factor"],
                    "input_types": ["raster"],
                    "output_type": "raster"
                }
            },
            "workflows": {
                "flood_risk": [
                    "elevation_analysis",
                    "slope_calculation", 
                    "drainage_analysis",
                    "buffer_water_bodies",
                    "overlay_analysis"
                ]
            }
        }
    
    def analyze_query(self, user_query: str) -> Dict[str, Any]:
        """
        Analyze user query to extract intent, entities, and requirements
        """
        # This would use NLP/LLM to parse the query
        analysis = {
            "intent": self._extract_intent(user_query),
            "entities": self._extract_entities(user_query),
            "spatial_extent": self._extract_spatial_extent(user_query),
            "data_requirements": self._extract_data_requirements(user_query)
        }
        
        reasoning = f"Query Analysis: Identified {analysis['intent']} task requiring {analysis['data_requirements']}"
        self.reasoning_history.append(reasoning)
        
        return analysis
    
    def _extract_intent(self, query: str) -> TaskType:
        """Extract the main geospatial task from query"""
        query_lower = query.lower()
        if "flood" in query_lower or "inundation" in query_lower:
            return TaskType.FLOOD_RISK
        elif "suitable" in query_lower or "optimal location" in query_lower:
            return TaskType.SITE_SUITABILITY
        elif "land cover" in query_lower or "classification" in query_lower:
            return TaskType.LAND_COVER
        else:
            return TaskType.BUFFER_ANALYSIS  # Default
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract geographical and domain entities"""
        # Simplified entity extraction
        entities = []
        if "river" in query.lower(): entities.append("river")
        if "elevation" in query.lower(): entities.append("elevation")
        if "road" in query.lower(): entities.append("road")
        return entities
    
    def _extract_spatial_extent(self, query: str) -> Optional[str]:
        """Extract spatial boundaries from query"""
        # This would use Named Entity Recognition for places
        return None
    
    def _extract_data_requirements(self, query: str) -> List[str]:
        """Determine what data sources are needed"""
        requirements = []
        entities = self._extract_entities(query)
        
        if "river" in entities: requirements.append("water_bodies")
        if "elevation" in entities: requirements.append("dem")
        if "road" in entities: requirements.append("road_network")
        
        return requirements
    
    def plan_workflow(self, query_analysis: Dict[str, Any]) -> List[GeoprocessingStep]:
        """
        Generate step-by-step workflow based on query analysis
        """
        task_type = query_analysis["intent"]
        entities = query_analysis["entities"]
        
        workflow_steps = []
        
        if task_type == TaskType.FLOOD_RISK:
            workflow_steps = self._plan_flood_risk_workflow(entities)
        elif task_type == TaskType.SITE_SUITABILITY:
            workflow_steps = self._plan_suitability_workflow(entities)
        
        # Add reasoning for workflow planning
        reasoning = f"Workflow Planning: Generated {len(workflow_steps)} steps for {task_type.value}"
        self.reasoning_history.append(reasoning)
        
        return workflow_steps
    
    def _plan_flood_risk_workflow(self, entities: List[str]) -> List[GeoprocessingStep]:
        """Plan specific workflow for flood risk assessment"""
        steps = []
        
        # Step 1: Load elevation data
        steps.append(GeoprocessingStep(
            step_id="step_1",
            operation="load_dem",
            parameters={"source": "dem_data"},
            input_data=["elevation_raster"],
            output_data="loaded_dem",
            reasoning="Loading elevation data is essential for flood modeling"
        ))
        
        # Step 2: Calculate slope
        steps.append(GeoprocessingStep(
            step_id="step_2", 
            operation="slope_analysis",
            parameters={"units": "degrees", "z_factor": 1.0},
            input_data=["loaded_dem"],
            output_data="slope_raster",
            reasoning="Slope analysis identifies areas prone to water accumulation",
            dependencies=["step_1"]
        ))
        
        # Step 3: Buffer water bodies
        if "river" in entities:
            steps.append(GeoprocessingStep(
                step_id="step_3",
                operation="buffer",
                parameters={"distance": 100, "units": "meters"},
                input_data=["water_bodies"],
                output_data="flood_zones",
                reasoning="Buffering water bodies identifies potential flood extent"
            ))
        
        return steps
    
    def _plan_suitability_workflow(self, entities: List[str]) -> List[GeoprocessingStep]:
        """Plan workflow for site suitability analysis"""
        # Implementation for suitability analysis
        return []
    
    def execute_workflow(self, steps: List[GeoprocessingStep], data_sources: Dict[str, Any]) -> WorkflowResult:
        """
        Execute the planned workflow steps
        """
        import time
        start_time = time.time()
        
        executed_steps = []
        outputs = {}
        errors = []
        
        try:
            for step in steps:
                # Check dependencies
                if step.dependencies:
                    for dep in step.dependencies:
                        if dep not in [s.step_id for s in executed_steps]:
                            raise Exception(f"Dependency {dep} not satisfied for {step.step_id}")
                
                # Execute step
                result = self._execute_step(step, data_sources, outputs)
                outputs[step.output_data] = result
                executed_steps.append(step)
                
                # Add execution reasoning
                reasoning = f"Executed {step.operation}: {step.reasoning}"
                self.reasoning_history.append(reasoning)
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"Workflow execution failed: {e}")
        
        execution_time = time.time() - start_time
        
        return WorkflowResult(
            workflow_id=f"workflow_{int(time.time())}",
            steps=executed_steps,
            outputs=outputs,
            reasoning_chain=self.reasoning_history.copy(),
            execution_time=execution_time,
            success=len(errors) == 0,
            error_messages=errors
        )
    
    def _execute_step(self, step: GeoprocessingStep, data_sources: Dict[str, Any], 
                     intermediate_outputs: Dict[str, Any]) -> Any:
        """Execute a single geoprocessing step"""
        
        if step.operation == "load_dem":
            # Mock loading DEM data
            return "dem_loaded_successfully"
        
        elif step.operation == "slope_analysis":
            # Mock slope calculation
            return "slope_raster_calculated"
            
        elif step.operation == "buffer":
            # Mock buffer operation
            return "buffer_zones_created"
        
        else:
            raise Exception(f"Unknown operation: {step.operation}")
    
    def generate_explanation(self, workflow_result: WorkflowResult) -> str:
        """Generate human-readable explanation of the workflow"""
        explanation = "## Geospatial Analysis Workflow Explanation\n\n"
        
        explanation += f"**Workflow ID:** {workflow_result.workflow_id}\n"
        explanation += f"**Execution Time:** {workflow_result.execution_time:.2f} seconds\n"
        explanation += f"**Status:** {'Success' if workflow_result.success else 'Failed'}\n\n"
        
        explanation += "### Chain of Thought Reasoning:\n"
        for i, reasoning in enumerate(workflow_result.reasoning_chain, 1):
            explanation += f"{i}. {reasoning}\n"
        
        explanation += "\n### Workflow Steps:\n"
        for step in workflow_result.steps:
            explanation += f"- **{step.operation}**: {step.reasoning}\n"
        
        if workflow_result.error_messages:
            explanation += "\n### Errors:\n"
            for error in workflow_result.error_messages:
                explanation += f"- {error}\n"
        
        return explanation

# Example usage
def main():
    """Example of how to use the GeoAI system"""
    
    # Initialize the reasoning engine
    geoai = GeoAIReasoningEngine()
    
    # Example query
    user_query = "I need to assess flood risk for areas near rivers with elevation data"
    
    # Analyze the query
    query_analysis = geoai.analyze_query(user_query)
    print("Query Analysis:", json.dumps(query_analysis, indent=2, default=str))
    
    # Plan workflow
    workflow_steps = geoai.plan_workflow(query_analysis)
    print(f"\nPlanned {len(workflow_steps)} workflow steps")
    
    # Mock data sources
    data_sources = {
        "elevation_raster": "path/to/dem.tif",
        "water_bodies": "path/to/rivers.shp"
    }
    
    # Execute workflow
    result = geoai.execute_workflow(workflow_steps, data_sources)
    
    # Generate explanation
    explanation = geoai.generate_explanation(result)
    print("\n" + explanation)

if __name__ == "__main__":
    main()