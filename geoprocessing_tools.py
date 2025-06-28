import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from typing import Dict, List, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class GeoprocessingTool(ABC):
    """Abstract base class for geoprocessing tools"""
    
    @abstractmethod
    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the geoprocessing operation"""
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate input parameters"""
        pass

class BufferTool(GeoprocessingTool):
    """Buffer analysis tool"""
    
    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        distance = parameters.get('distance', 1000)
        input_data = inputs.get('geometry')
        
        if isinstance(input_data, gpd.GeoDataFrame):
            buffered = input_data.copy()
            buffered['geometry'] = input_data.geometry.buffer(distance)
            return {'result': buffered}
        
        return {'result': None, 'error': 'Invalid input data'}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return 'distance' in parameters and isinstance(parameters['distance'], (int, float))

class OverlayTool(GeoprocessingTool):
    """Spatial overlay operations"""
    
    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        operation = parameters.get('operation', 'intersection')
        layer1 = inputs.get('layer1')
        layer2 = inputs.get('layer2')
        
        if not isinstance(layer1, gpd.GeoDataFrame) or not isinstance(layer2, gpd.GeoDataFrame):
            return {'result': None, 'error': 'Invalid input layers'}
        
        try:
            if operation == 'intersection':
                result = gpd.overlay(layer1, layer2, how='intersection')
            elif operation == 'union':
                result = gpd.overlay(layer1, layer2, how='union')
            elif operation == 'difference':
                result = gpd.overlay(layer1, layer2, how='difference')
            else:
                return {'result': None, 'error': f'Unknown operation: {operation}'}
            
            return {'result': result}
        except Exception as e:
            return {'result': None, 'error': str(e)}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        valid_ops = ['intersection', 'union', 'difference']
        return parameters.get('operation', 'intersection') in valid_ops

class ClipTool(GeoprocessingTool):
    """Clip geometries by boundary"""
    
    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_layer = inputs.get('input_layer')
        clip_layer = inputs.get('clip_layer')
        
        if not isinstance(input_layer, gpd.GeoDataFrame) or not isinstance(clip_layer, gpd.GeoDataFrame):
            return {'result': None, 'error': 'Invalid input layers'}
        
        try:
            clipped = gpd.clip(input_layer, clip_layer)
            return {'result': clipped}
        except Exception as e:
            return {'result': None, 'error': str(e)}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return True

class DistanceAnalysisTool(GeoprocessingTool):
    """Distance analysis operations"""
    
    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        operation = parameters.get('operation', 'nearest')
        source_layer = inputs.get('source_layer')
        target_layer = inputs.get('target_layer')
        
        if not isinstance(source_layer, gpd.GeoDataFrame):
            return {'result': None, 'error': 'Invalid source layer'}
        
        try:
            if operation == 'nearest':
                # Calculate distance to nearest feature
                if isinstance(target_layer, gpd.GeoDataFrame):
                    distances = []
                    for geom in source_layer.geometry:
                        min_dist = min([geom.distance(target_geom) for target_geom in target_layer.geometry])
                        distances.append(min_dist)
                    
                    result = source_layer.copy()
                    result['distance'] = distances
                    return {'result': result}
            
            return {'result': None, 'error': f'Unknown operation: {operation}'}
        except Exception as e:
            return {'result': None, 'error': str(e)}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return parameters.get('operation', 'nearest') in ['nearest']

class SlopeAnalysisTool(GeoprocessingTool):
    """Slope analysis from elevation data"""
    
    def execute(self, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        elevation_data = inputs.get('elevation_data')
        threshold = parameters.get('threshold', 15.0)
        
        # Simplified slope calculation - in real implementation would use rasterio/GDAL
        try:
            # Mock slope calculation
            if isinstance(elevation_data, gpd.GeoDataFrame):
                result = elevation_data.copy()
                # Simulate slope values
                result['slope'] = np.random.uniform(0, 30, len(elevation_data))
                result['steep'] = result['slope'] > threshold
                return {'result': result}
            
            return {'result': None, 'error': 'Invalid elevation data'}
        except Exception as e:
            return {'result': None, 'error': str(e)}
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return isinstance(parameters.get('threshold', 15.0), (int, float))

class GeoprocessingToolkit:
    """Main toolkit class that orchestrates all geoprocessing operations"""
    
    def __init__(self):
        self.tools = {
            'buffer': BufferTool(),
            'overlay': OverlayTool(),
            'clip': ClipTool(),
            'distance': DistanceAnalysisTool(),
            'slope': SlopeAnalysisTool()
        }
        logger.info("GeoprocessingToolkit initialized")
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tools"""
        return list(self.tools.keys())
    
    def execute_operation(self, operation: str, parameters: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a geoprocessing operation"""
        if operation not in self.tools:
            return {'result': None, 'error': f'Unknown operation: {operation}'}
        
        tool = self.tools[operation]
        
        # Validate parameters
        if not tool.validate_parameters(parameters):
            return {'result': None, 'error': 'Invalid parameters'}
        
        # Execute operation
        try:
            result = tool.execute(parameters, inputs)
            logger.info(f"Successfully executed {operation}")
            return result
        except Exception as e:
            logger.error(f"Error executing {operation}: {str(e)}")
            return {'result': None, 'error': str(e)}
    
    def create_sample_data(self, data_type: str = 'points') -> gpd.GeoDataFrame:
        """Create sample geospatial data for testing"""
        if data_type == 'points':
            # Create sample points around Maharashtra
            coords = [
                (75.7139, 19.7515),  # Maharashtra center
                (72.8777, 19.0760),  # Mumbai
                (73.8567, 18.5204),  # Pune
                (79.0882, 21.1458),  # Nagpur
            ]
            
            geometries = [Point(lon, lat) for lon, lat in coords]
            data = {
                'name': ['Center', 'Mumbai', 'Pune', 'Nagpur'],
                'population': [1000000, 12442373, 3124458, 2405421],
                'geometry': geometries
            }
            
            return gpd.GeoDataFrame(data, crs='EPSG:4326')
        
        elif data_type == 'polygons':
            # Create sample polygons
            poly1 = Polygon([(75.0, 19.0), (76.0, 19.0), (76.0, 20.0), (75.0, 20.0)])
            poly2 = Polygon([(72.5, 18.5), (73.5, 18.5), (73.5, 19.5), (72.5, 19.5)])
            
            data = {
                'name': ['Region1', 'Region2'],
                'area_type': ['urban', 'rural'],
                'geometry': [poly1, poly2]
            }
            
            return gpd.GeoDataFrame(data, crs='EPSG:4326')
        
        return gpd.GeoDataFrame()
    
    def validate_workflow_step(self, step: Dict[str, Any]) -> bool:
        """Validate a workflow step"""
        required_fields = ['operation', 'parameters']
        
        for field in required_fields:
            if field not in step:
                return False
        
        operation = step['operation']
        if operation not in self.tools:
            return False
        
        return self.tools[operation].validate_parameters(step['parameters'])
    
    def execute_workflow(self, workflow_steps: List[Dict[str, Any]], initial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete workflow"""
        current_data = initial_data.copy()
        results = []
        
        for i, step in enumerate(workflow_steps):
            if not self.validate_workflow_step(step):
                return {
                    'success': False,
                    'error': f'Invalid workflow step {i+1}',
                    'results': results
                }
            
            # Execute step
            step_result = self.execute_operation(
                step['operation'],
                step['parameters'],
                current_data
            )
            
            if step_result.get('error'):
                return {
                    'success': False,
                    'error': f'Step {i+1} failed: {step_result["error"]}',
                    'results': results
                }
            
            # Update current data with result
            if step_result.get('result') is not None:
                output_name = step.get('output', f'step_{i+1}_result')
                current_data[output_name] = step_result['result']
            
            results.append({
                'step': i+1,
                'operation': step['operation'],
                'success': True,
                'output': step.get('output', f'step_{i+1}_result')
            })
        
        return {
            'success': True,
            'results': results,
            'final_data': current_data
        }

# Utility functions for backward compatibility
def calculate_area(length, width):
    """Calculate the area of a rectangle."""
    return length * width

def calculate_perimeter(length, width):
    """Calculate the perimeter of a rectangle."""
    return 2 * (length + width)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Earth radius in kilometers

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance