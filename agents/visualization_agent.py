# agents/visualization_agent.py
import plotly.express as px
import streamlit as st
import pandas as pd
from io import StringIO

# Function to return the file content for download
def get_plotly_figure_html(fig):
    """Saves the Plotly figure to an in-memory buffer as HTML."""
    buffer = StringIO()
    fig.write_html(buffer, include_plotlyjs="cdn")
    return buffer.getvalue().encode('utf-8')

def visualization_agent(df, plot_style="line_depth"):
    """
    Draws typical ocean plots in various styles and returns a dictionary 
    of generated figures for display and download.
    
    Assumes incoming DataFrame has columns renamed to: depth, temperature, salinity.
    """
    if df is None or df.empty:
        return {}

    figures = {}

    # Helper function to generate and display a plot
    def generate_plot(x_col, y_col, title, x_label, y_label, style):
        # Check if necessary columns exist for the plot
        if x_col not in df.columns or (y_col and y_col not in df.columns):
            return None

        # --- Plot Style Implementation ---
        if style == "line_depth":
            fig = px.line(df, x=x_col, y=y_col, title=title, labels={x_col: x_label, y_col: y_label})
            if y_col == "depth":
                fig.update_yaxes(autorange="reversed")
        
        elif style == "scatter_map":
            if "latitude" not in df.columns or "longitude" not in df.columns:
                 return None
            
            fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_data=df.columns.tolist(), zoom=1)
            fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            title = "Geographic Profile Locations"
            
        elif style == "scatter_profile":
            fig = px.scatter(df, x=x_col, y=y_col, title=title, labels={x_col: x_label, y_col: y_label})
            if y_col == "depth":
                fig.update_yaxes(autorange="reversed")
        
        elif style == "histogram":
            # For visualizing distribution of a single variable (Uses x_col)
            fig = px.histogram(df, x=x_col, title=f"Distribution of {x_label}", labels={x_col: x_label})
            
        elif style == "box_plot":
            # For visualizing distribution summary (Uses x_col)
            fig = px.box(df, y=x_col, title=f"Box Plot of {x_label}", labels={x_col: x_label})

        else:
             # Default to line plot
            fig = px.line(df, x=x_col, y=y_col, title=title, labels={x_col: x_label, y_col: y_label})
            if y_col == "depth":
                fig.update_yaxes(autorange="reversed")
            
        fig.update_layout(title_text=title)
        st.subheader(title)
        st.plotly_chart(fig, use_container_width=True)
        return fig
        # --- End Plot Style Implementation ---

    # --- Generate Plots based on available data and selected style ---
    
    # 1. Depth Profile Plots (If depth and T/S data are present)
    if "temperature" in df.columns and "depth" in df.columns:
        # Pass the depth and temperature columns to generate a plot in the selected style
        temp_fig = generate_plot("temperature", "depth", "Temperature Profile", "Temp (°C)", "Depth (dbar)", plot_style)
        if temp_fig:
            figures["Temperature_Profile"] = temp_fig

    if "salinity" in df.columns and "depth" in df.columns:
        # Pass the depth and salinity columns to generate a plot in the selected style
        salt_fig = generate_plot("salinity", "depth", "Salinity Profile", "Salinity (PSU)", "Depth (dbar)", plot_style)
        if salt_fig:
            figures["Salinity_Profile"] = salt_fig
            
    # 3. Map/Location plot (If geo data is present, always drawn as scatter map if selected)
    if plot_style == "scatter_map" and "latitude" in df.columns and "longitude" in df.columns:
        # Note: Map is always generated if data exists and is explicitly named "scatter_map"
        map_fig = generate_plot("longitude", "latitude", "Map of Profile Locations", "Longitude", "Latitude", "scatter_map")
        if map_fig:
            figures["Map_Location"] = map_fig

    # If the user selected a single variable plot (histogram or box_plot) and we haven't plotted one yet,
    # default to using the primary measurement column (temperature)
    if plot_style in ["histogram", "box_plot"] and not figures and "temperature" in df.columns:
        single_fig = generate_plot("temperature", None, "Measurement Distribution", "Temp (°C)", None, plot_style)
        if single_fig:
            figures["Distribution_Plot"] = single_fig

    return figures
