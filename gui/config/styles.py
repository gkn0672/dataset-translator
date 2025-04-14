CSS = """
/* Dark theme base colors */
body {
    background-color: #1f1f1f;
    color: #f0f0f0;
}

/* Log output styling */
#logs-output {
    background-color: #2d2d2d;
    font-family: monospace;
    color: #e0e0e0;
    border: 1px solid #444;
    border-radius: 6px;
}

/* JSON Editor styling */
#json-editor {
    background-color: #2d2d2d;
    border-radius: 6px;
    border: 1px solid #444;
}

/* Button styling */
#fetch-btn {
    min-width: 150px;
}

#start-translation-btn {
    width: 100%;
    margin-top: 20px;
    margin-bottom: 20px;
    border-radius: 8px;
    background-color: #ff7f00;
    color: white;
    font-weight: bold;
}

/* Group styling to mimic boxes */
.gradio-group {
    border: 1px solid #444;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
    background-color: #2d2d2d;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

/* Dropdown styling */
select, .gr-dropdown {
    background-color: #3a3a3a !important;
    border: 1px solid #555 !important;
    color: #e0e0e0 !important;
    border-radius: 4px !important;
}

/* Input styling */
input[type="text"], input[type="number"], textarea {
    background-color: #3a3a3a !important;
    border: 1px solid #555 !important;
    color: #e0e0e0 !important;
    border-radius: 4px !important;
}

/* Additional fields table styling */
.additional-fields-table table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 15px;
}

.additional-fields-table th {
    background-color: #444;
    padding: 8px;
    text-align: left;
    color: #e0e0e0;
    border-radius: 4px 4px 0 0;
}

.additional-fields-table td {
    padding: 8px;
    border-bottom: 1px solid #444;
    color: #e0e0e0;
}

/* Tab styling */
.tabs {
    margin-top: 10px;
}

/* Tabs container */
.gradio-tabs {
    border-bottom: 1px solid #444;
}

/* Tab buttons */
.gradio-tabs button {
    color: #ccc;
    background-color: transparent;
    border: none;
    padding: 8px 16px;
    margin-right: 4px;
    border-radius: 4px 4px 0 0;
}

/* Active tab */
.gradio-tabs button.selected {
    color: #ff7f00;
    background-color: #2d2d2d;
    border: 1px solid #444;
    border-bottom: none;
    font-weight: bold;
}

/* Tab content */
.gradio-tabitem {
    padding: 15px 0;
}
"""
