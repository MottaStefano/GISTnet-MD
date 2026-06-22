namespace eval ::xAIDashboard {}

proc ::xAIDashboard::load_csv {filepath} {
    if {![file exists $filepath]} return

    array unset ::xAIDashboard::time_data
    set ::xAIDashboard::pos_cols {}
    set ::xAIDashboard::max_sal 0.0001
    set ::xAIDashboard::available_metrics {}

    set fp [open $filepath r]
    set header [gets $fp]
    set cols [split $header ","]

    set metric_indices {}

    # Identifica le colonne delle "Track" e le colonne delle altre Metriche
    for {set i 0} {$i < [llength $cols]} {incr i} {
        set colname [string trim [lindex $cols $i]]
        if {[string match "Window_Track_*" $colname]} {
            if {[llength $::xAIDashboard::pos_cols] < 8} { lappend ::xAIDashboard::pos_cols $i }
        } elseif {$colname ne "Frame" && $colname ne ""} {
            lappend ::xAIDashboard::available_metrics $colname
            lappend metric_indices $i
        }
    }

    # Setup Metrica di Default (Forza Confidence se presente)
    if {[llength $::xAIDashboard::available_metrics] == 0} {
        lappend ::xAIDashboard::available_metrics "Confidence"
    }
    if {[lsearch -exact $::xAIDashboard::available_metrics "Confidence"] != -1} {
        set ::xAIDashboard::top_metric "Confidence"
    } else {
        set ::xAIDashboard::top_metric [lindex $::xAIDashboard::available_metrics 0]
    }

    # Aggiorna il menu a tendina nell'interfaccia con i valori veri trovati nel CSV
    catch { .xaigui.main.p3d.opt_metric configure -values $::xAIDashboard::available_metrics }

    set frame_count 0
    while {[gets $fp line] >= 0} {
        set vals [split $line ","]
        if {[llength $vals] < 2} continue

        # Estrazione Dati Saliency (Overlapping Windows)
        set frame_sals {}
        foreach col_idx $::xAIDashboard::pos_cols {
            set val [string trim [lindex $vals $col_idx]]
            if {$val eq "NaN" || $val eq ""} { set val "NaN" } else {
                if {$val > $::xAIDashboard::max_sal} { set ::xAIDashboard::max_sal $val }
            }
            lappend frame_sals $val
        }
        set ::xAIDashboard::time_data($frame_count,sals) $frame_sals

        # Estrazione Dinamica Metriche Selezionabili (Directional, Confidence, Anomaly_Score ecc.)
        foreach idx $metric_indices metric_name $::xAIDashboard::available_metrics {
            set val [string trim [lindex $vals $idx]]
            if {![string is double -strict $val]} { set val 0.0 }
            set ::xAIDashboard::time_data($frame_count,$metric_name) $val
        }

        # Fallback se non ci sono metriche
        if {[llength $metric_indices] == 0} {
            set ::xAIDashboard::time_data($frame_count,$::xAIDashboard::top_metric) 0.0
        }

        incr frame_count
    }
    close $fp
    set ::xAIDashboard::num_frames $frame_count

    ::xAIDashboard::build_dynamic_checkboxes
    ::xAIDashboard::draw_plots
}

proc ::xAIDashboard::load_dat {filepath} {
    if {![file exists $filepath]} return

    array unset ::xAIDashboard::spat_data
    set ::xAIDashboard::max_spat -9999.0
    set ::xAIDashboard::min_spat 9999.0
    set ::xAIDashboard::num_res 0

    set fp [open $filepath r]
    set file_data [read $fp]
    close $fp

    set frame_count 0
    foreach line [split $file_data "\n"] {
        set line [string trim $line]
        if {$line == ""} continue

        if {[string index $line 0] == "#"} {
            set meta_parts [split $line " "]
            if {[llength $meta_parts] > 3} { set ::xAIDashboard::num_res [lindex $meta_parts 3] }
            continue
        }

        set vals [split $line " "]
        if {$::xAIDashboard::num_res == 0} { set ::xAIDashboard::num_res [llength $vals] }

        set ::xAIDashboard::spat_data($frame_count) $vals
        foreach v $vals {
            if {$v ne ""} {
                if {$v > $::xAIDashboard::max_spat} { set ::xAIDashboard::max_spat $v }
                if {$v < $::xAIDashboard::min_spat} { set ::xAIDashboard::min_spat $v }
            }
        }
        incr frame_count
    }

    # Prevenzione per dati malformati o zero variance
    if {$::xAIDashboard::max_spat < $::xAIDashboard::min_spat} {
        set ::xAIDashboard::max_spat 0.0001
        set ::xAIDashboard::min_spat 0.0
    }

    ::xAIDashboard::setup_vmd_trace
    ::xAIDashboard::draw_plots
}
