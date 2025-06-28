import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from geoai_framework import GeoAIReasoningEngine, TaskType
import json

class TestGeoAISystem:
    
    def setup_method(self):
        """Setup test environment"""
        self.geoai = GeoAIReasoningEngine()
    
    def test_query_analysis(self):
        """Test query analysis functionality"""
        query = "Assess flood risk for areas near rivers"
        analysis = self.geoai.analyze_query(query)
        
        assert analysis["intent"] == TaskType.FLOOD_RISK
        assert "river" in analysis["entities"]
        assert "water_bodies" in analysis["data_requirements"]
    
    def test_workflow_planning(self):
        """Test workflow planning"""
        query_analysis = {
            "intent": TaskType.FLOOD_RISK,
            "entities": ["river", "elevation"],
            "data_requirements": ["water_bodies", "dem"]
        }
        
        workflow = self.geoai.plan_workflow(query_analysis)
        
        assert len(workflow) > 0
        assert any("load_dem" in step.operation for step in workflow)
        assert any("slope" in step.operation for step in workflow)
    
    def test_reasoning_chain(self):
        """Test chain-of-thought reasoning"""
        query = "Find suitable locations for solar farms"
        analysis = self.geoai.analyze_query(query)
        workflow = self.geoai.plan_workflow(analysis)
        
        assert len(self.geoai.reasoning_history) > 0
        assert any("Query Analysis" in reasoning for reasoning in self.geoai.reasoning_history)
    
    def test_workflow_execution(self):
        """Test workflow execution (mocked)"""
        steps = [
            {
                "step_id": "step_1",
                "operation": "load_dem",
                "parameters": {"source": "test_dem"},
                "input_data": ["elevation"],
                "output_data": "loaded_dem",
                "reasoning": "Test step"
            }
        ]
        
        data_sources = {"elevation": "test_data"}
        # This would test actual execution in a real implementation
        
    def test_error_handling(self):
        """Test error handling capabilities"""
        # Test with invalid query
        query = ""
        analysis = self.geoai.analyze_query(query)
        
        # Should handle gracefully
        assert "intent" in analysis

if __name__ == "__main__":
    pytest.main([__file__])