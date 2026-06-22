# =============================================================================
# xAI VMD Dashboard - Modern Modular Version (Triple Graph + Zoom + 3D)
# =============================================================================

package require Tk
catch {package require ttk}

namespace eval ::xAIDashboard {
    variable script_dir [file normalize [file dirname [info script]]]
    if {$script_dir eq ""} { set script_dir "." }

    # Variabili Globali Dati
    variable num_frames 0
    variable num_res 0
    variable active_mol -1

    # Arrays temporali e spaziali
    variable time_data
    variable spat_data
    array set time_data {}
    array set spat_data {}

    # Limiti calcolati dinamicamente
    variable max_sal 0.0001
    variable max_spat 0.0001
    variable min_spat 0.0

    # Metriche Plot Superiore
    variable top_metric "Confidence"
    variable available_metrics {"Confidence" "Directional" "Anomaly_Score"}

    # Gestione delle posizioni Saliency (Max 8)
    variable pos_cols {}
    variable show_pos
    array set show_pos {}
    for {set i 0} {$i < 8} {incr i} { set show_pos($i) 1 }

    # Colori ottimizzati per SFONDO CHIARO (Tracce e grafici)
    variable pos_colors {"#0055CC" "#CC00CC" "#1C991C" "#D95E00" "#B38F00" "#D11141" "#00AEDB" "#333333"}

    # Opzioni 3D e Zoom
    variable zoom_window 100
    variable atom_sel "name CA"
    variable apply_all 1
    variable rep_style "NewCartoon"

    # Variabili VMD
    variable trace_active 0
}

# Caricamento Moduli
source [file join $::xAIDashboard::script_dir "src" "data.tcl"]
source [file join $::xAIDashboard::script_dir "src" "plot.tcl"]
source [file join $::xAIDashboard::script_dir "src" "vmd_update.tcl"]
source [file join $::xAIDashboard::script_dir "src" "ui.tcl"]

# Avvio Interfaccia
::xAIDashboard::init_ui
