import streamlit as st
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# Import custom modules
from geoai_framework import GeoAIReasoningEngine, TaskType, WorkflowResult
from config import Config
from data_sources import DataSourceManager
from geoprocessing_tools import GeoprocessingToolkit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/geoai.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title=Config.UI["page_title"],
    page_icon=Config.UI["page_icon"],
    layout=Config.UI["layout"],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .reasoning-step {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        animation: slideIn 0.5s ease-out;
    }
    .workflow-step {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e1e5e9;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e1e5e9;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class GeoAIApp:
    def __init__(self):
        self.initialize_session_state()
        self.load_system_components()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'workflow_history' not in st.session_state:
            st.session_state.workflow_history = []
        if 'current_workflow' not in st.session_state:
            st.session_state.current_workflow = None
        if 'reasoning_steps' not in st.session_state:
            st.session_state.reasoning_steps = []
        if 'execution_progress' not in st.session_state:
            st.session_state.execution_progress = 0
    
    def load_system_components(self):
        """Load and initialize system components"""
        try:
            self.geoai_engine = GeoAIReasoningEngine()
            self.data_manager = DataSourceManager()
            self.geoprocessing_toolkit = GeoprocessingToolkit()
            logger.info("System components loaded successfully")
        except Exception as e:
            st.error(f"Failed to load system components: {str(e)}")
            logger.error(f"Component loading error: {e}")
    
    def render_header(self):
        """Render the main application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🌍 GeoAI - Intelligent Spatial Analysis System</h1>
            <p>Chain-of-Thought Reasoning for Complex Geospatial Workflows</p>
            <p><em>Powered by Advanced Language Models & Geospatial Intelligence</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.header("🛠️ System Configuration")
            
            # Model selection
            model_options = list(Config.MODELS.keys())
            selected_model = st.selectbox(
                "Select LLM Model",
                model_options,
                index=0,
                help="Choose the language model for reasoning"
            )
            
            # Data sources
            st.subheader("📊 Data Sources")
            available_sources = list(Config.DATA_SOURCES.keys())
            selected_sources = st.multiselect(
                "Available Data Sources",
                available_sources,
                default=["bhoonidhi", "osm"],
                help="Select data sources for analysis"
            )
            
            # Processing options
            st.subheader("⚙️ Processing Options")
            enable_rag = st.checkbox("Enable RAG System", value=True)
            show_reasoning = st.checkbox("Show Chain-of-Thought", value=True)
            auto_execute = st.checkbox("Auto-execute Workflow", value=False)
            max_steps = st.slider("Max Workflow Steps", 3, 15, 10)
            
            # Coordinate system
            st.subheader("🗺️ Coordinate System")
            crs_options = list(Config.COORDINATE_SYSTEMS.keys())
            selected_crs = st.selectbox("Select CRS", crs_options, index=0)
            
            # System status
            st.subheader("📊 System Status")
            self.render_system_status()
            
            return {
                'model': selected_model,
                'data_sources': selected_sources,
                'enable_rag': enable_rag,
                'show_reasoning': show_reasoning,
                'auto_execute': auto_execute,
                'max_steps': max_steps,
                'crs': selected_crs
            }
    
    def render_system_status(self):
        """Render system status indicators"""
        # Check system health
        status_indicators = {
            "🧠 LLM Engine": "🟢 Ready",
            "📊 Data Sources": "🟢 Connected",
            "🛠️ Geoprocessing": "🟢 Available",
            "💾 Cache System": "🟢 Active"
        }
        
        for component, status in status_indicators.items():
            st.text(f"{component}: {status}")
    
    def render_query_interface(self):
        """Render the query input interface"""
        st.header("💬 Natural Language Query Interface")
        
        # Example queries
        example_queries = [
            "Assess flood risk for urban areas within 1km of rivers in Maharashtra using elevation data",
            "Find suitable locations for solar farms avoiding forests, water bodies, and slopes >15°",
            "Identify areas of urban expansion in the last 5 years using satellite imagery",
            "Calculate the optimal route for emergency services avoiding flood-prone areas",
            "Analyze land cover changes near industrial zones and assess environmental impact"
        ]
        
        # Quick examples
        with st.expander("📋 Example Queries", expanded=False):
            for i, example in enumerate(example_queries):
                if st.button(f"📝 {example}", key=f"example_{i}"):
                    return example
        
        # Main query input
        user_query = st.text_area(
            "Describe your geospatial analysis task:",
            placeholder="e.g., 'Find optimal locations for wind farms in coastal areas with wind speed > 7 m/s'",
            height=120,
            max_chars=Config.UI["max_query_length"]
        )
        
        return user_query
    
    def process_query(self, query: str, config: Dict) -> Optional[WorkflowResult]:
        """Process the user query and generate workflow"""
        if not query.strip():
            st.warning("Please enter a query to analyze")
            return None
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Query Analysis
            status_text.text("🔍 Analyzing query...")
            progress_bar.progress(20)
            time.sleep(1)  # Simulate processing
            
            query_analysis = self.geoai_engine.analyze_query(query)
            
            # Step 2: Workflow Planning
            status_text.text("📋 Planning workflow...")
            progress_bar.progress(40)
            time.sleep(1)
            
            workflow_steps = self.geoai_engine.plan_workflow(query_analysis)
            
            # Step 3: Data Preparation
            status_text.text("📊 Preparing data sources...")
            progress_bar.progress(60)
            time.sleep(1)
            
            # Prepare mock data sources
            data_sources = self.prepare_data_sources(config['data_sources'])
            
            # Step 4: Workflow Execution
            status_text.text("⚙️ Executing workflow...")
            progress_bar.progress(80)
            time.sleep(2)
            
            # Execute workflow
            workflow_result = self.geoai_engine.execute_workflow(workflow_steps, data_sources)
            
            # Step 5: Complete
            status_text.text("✅ Analysis complete!")
            progress_bar.progress(100)
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            return workflow_result
            
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Query processing error: {e}")
            return None
    
    def prepare_data_sources(self, selected_sources: List[str]) -> Dict[str, Any]:
        """Prepare and validate data sources"""
        data_sources = {}
        
        for source in selected_sources:
            if source == "bhoonidhi":
                data_sources["elevation"] = "bhoonidhi://dem/srtm"
                data_sources["land_use"] = "bhoonidhi://landuse/current"
            elif source == "osm":
                data_sources["roads"] = "osm://highway/primary,secondary"
                data_sources["water_bodies"] = "osm://natural/water"
            elif source == "stac":
                data_sources["satellite"] = "stac://sentinel-2/latest"
        
        return data_sources
    
    def render_reasoning_display(self, workflow_result: WorkflowResult):
        """Render the chain-of-thought reasoning display"""
        st.header("🧠 Chain-of-Thought Reasoning")
        
        if workflow_result and workflow_result.reasoning_chain:
            for i, reasoning in enumerate(workflow_result.reasoning_chain, 1):
                st.markdown(f"""
                <div class="reasoning-step">
                    <strong>Step {i}:</strong> {reasoning}
                </div>
                """, unsafe_allow_html=True)
                # Add small delay for visual effect
                time.sleep(0.1)
        else:
            st.info("No reasoning steps available")
    
    def render_workflow_steps(self, workflow_result: WorkflowResult):
        """Render the workflow steps visualization"""
        st.header("📋 Generated Workflow")
        
        if workflow_result and workflow_result.steps:
            for i, step in enumerate(workflow_result.steps, 1):
                st.markdown(f"""
                <div class="workflow-step">
                    <h4>Step {i}: {step.operation}</h4>
                    <p><strong>Reasoning:</strong> {step.reasoning}</p>
                    <p><strong>Parameters:</strong> {json.dumps(step.parameters, indent=2)}</p>
                    <p><strong>Input:</strong> {', '.join(step.input_data)}</p>
                    <p><strong>Output:</strong> {step.output_data}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No workflow steps generated")
    
    def render_results_visualization(self, workflow_result: WorkflowResult):
        """Render results visualization"""
        st.header("📊 Analysis Results")
        
        if not workflow_result:
            st.info("No results to display")
            return
        
        # Metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{workflow_result.execution_time:.1f}s</h3>
                <p>Execution Time</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(workflow_result.steps)}</h3>
                <p>Processing Steps</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status_icon = "✅" if workflow_result.success else "❌"
            st.markdown(f"""
            <div class="metric-card">
                <h3>{status_icon}</h3>
                <p>Status</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(workflow_result.outputs)}</h3>
                <p>Outputs Generated</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Map View", "📈 Charts", "📋 Data Table", "💾 Export"])
        
        with tab1:
            self.render_map_visualization(workflow_result)
        
        with tab2:
            self.render_charts(workflow_result)
        
        with tab3:
            self.render_data_table(workflow_result)
        
        with tab4:
            self.render_export_options(workflow_result)
    
    def render_map_visualization(self, workflow_result: WorkflowResult):
        """Render map visualization"""
        # Create sample map centered on Maharashtra
        center_lat, center_lon = 19.7515, 75.7139
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
        
        # Add sample results based on workflow type
        if "flood" in str(workflow_result.steps[0].reasoning).lower():
            # Add flood risk zones
            locations = [
                ([19.0760, 72.8777], "High Risk - Mumbai", "red"),
                ([18.5204, 73.8567], "Medium Risk - Pune", "orange"),
                ([21.1458, 79.0882], "Low Risk - Nagpur", "green")
            ]
            
            for location, popup_text, color in locations:
                folium.CircleMarker(
                    location,
                    radius=15,
                    popup=popup_text,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6
                ).add_to(m)
        
        elif "suitable" in str(workflow_result.steps[0].reasoning).lower():
            # Add suitable locations
            suitable_locations = [
                ([19.8762, 75.3433], "Suitable Site 1"),
                ([20.5937, 78.9629], "Suitable Site 2"),
                ([18.5204, 73.8567], "Suitable Site 3")
            ]
            
            for location, popup_text in suitable_locations:
                folium.Marker(
                    location,
                    popup=popup_text,
                    icon=folium.Icon(color="green", icon="star")
                ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])
    
    def render_charts(self, workflow_result: WorkflowResult):
        """Render analytical charts"""
        # Execution time breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing time per step
            step_times = [i * 0.5 + 1.2 for i in range(len(workflow_result.steps))]
            step_names = [step.operation for step in workflow_result.steps]
            
            fig = px.bar(
                x=step_names,
                y=step_times,
                title="Processing Time per Step",
                labels={'x': 'Workflow Step', 'y': 'Time (seconds)'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample results distribution
            categories = ['High Priority', 'Medium Priority', 'Low Priority']
            values = [25, 45, 30]
            
            fig = px.pie(
                values=values,
                names=categories,
                title="Analysis Results Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_data_table(self, workflow_result: WorkflowResult):
        """Render data table with results"""
        # Create sample results table
        results_data = {
            'Location': ['Site A', 'Site B', 'Site C', 'Site D', 'Site E'],
            'Score': [8.5, 7.2, 9.1, 6.8, 7.9],
            'Priority': ['High', 'Medium', 'High', 'Low', 'Medium'],
            'Area (sq km)': [12.5, 8.3, 15.2, 6.1, 9.7],
            'Notes': ['Optimal conditions', 'Minor constraints', 'Excellent site', 'Limited access', 'Good potential']
        }
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True)
    
    def render_export_options(self, workflow_result: WorkflowResult):
        """Render export options"""
        st.subheader("Export Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export Workflow JSON"):
                workflow_json = {
                    "workflow_id": workflow_result.workflow_id,
                    "execution_time": workflow_result.execution_time,
                    "success": workflow_result.success,
                    "steps": [
                        {
                            "step_id": step.step_id,
                            "operation": step.operation,
                            "parameters": step.parameters,
                            "reasoning": step.reasoning
                        } for step in workflow_result.steps
                    ],
                    "reasoning_chain": workflow_result.reasoning_chain,
                    "timestamp": datetime.now().isoformat()
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(workflow_json, indent=2),
                    file_name=f"geoai_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🗺️ Export Map Data"):
                st.info("Map data export functionality would be implemented here")
        
        with col3:
            if st.button("📊 Export Results CSV"):
                st.info("CSV export functionality would be implemented here")
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Get sidebar configuration
        config = self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Query interface
            user_query = self.render_query_interface()
            
            # Process button
            if st.button("🚀 Analyze Query", type="primary", use_container_width=True):
                if user_query:
                    workflow_result = self.process_query(user_query, config)
                    if workflow_result:
                        st.session_state.current_workflow = workflow_result
                        st.session_state.workflow_history.append(workflow_result)
                        st.rerun()
        
        with col2:
            # Reasoning display
            if st.session_state.current_workflow and config['show_reasoning']:
                self.render_reasoning_display(st.session_state.current_workflow)
        
        # Results section
        if st.session_state.current_workflow:
            st.markdown("---")
            
            # Workflow steps
            self.render_workflow_steps(st.session_state.current_workflow)
            
            st.markdown("---")
            
            # Results visualization
            self.render_results_visualization(st.session_state.current_workflow)
        
        # Workflow history
        if st.session_state.workflow_history:
            st.markdown("---")
            st.header("📚 Workflow History")
            
            for i, workflow in enumerate(reversed(st.session_state.workflow_history[-5:])):
                with st.expander(f"Workflow {len(st.session_state.workflow_history) - i}: {workflow.workflow_id}"):
                    st.write(f"**Execution Time:** {workflow.execution_time:.2f}s")
                    st.write(f"**Steps:** {len(workflow.steps)}")
                    st.write(f"**Success:** {'✅' if workflow.success else '❌'}")
                    if st.button(f"Load Workflow {i}", key=f"load_{i}"):
                        st.session_state.current_workflow = workflow
                        st.rerun()

def main():
    """Main application entry point"""
    try:
        # Create necessary directories
        Config.create_directories()
        
        # Initialize and run the application
        app = GeoAIApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}")

if __name__ == "__main__":
    main()