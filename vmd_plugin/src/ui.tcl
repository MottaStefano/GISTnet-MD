namespace eval ::xAIDashboard {}

proc ::xAIDashboard::init_ui {} {
    if {[winfo exists .xaigui]} { destroy .xaigui }
    toplevel .xaigui
    wm title .xaigui "MD-GNN xAI Dashboard"
    catch {ttk::style theme use clam}

    # Stile per il pulsante in evidenza
    ttk::style configure "Accent.TButton" -foreground "#0055CC" -font "Helvetica 10 bold"

    set main [ttk::frame .xaigui.main -padding "10 10 10 10"]
    pack $main -fill both -expand 1

    # --- IO Panel ---
    set f_io [ttk::labelframe $main.io -text " 1. Data Input " -padding "10 5"]
    pack $f_io -fill x -pady "0 5"

    ttk::button $f_io.btn_dat -text "Load Spatial (.dat)" -command {
        set file [tk_getOpenFile -title "Select .dat file"]
        if {$file ne ""} { ::xAIDashboard::load_dat $file }
    }
    ttk::button $f_io.btn_csv -text "Load Temporal (.csv)" -command {
        set file [tk_getOpenFile -title "Select .csv file"]
        if {$file ne ""} { ::xAIDashboard::load_csv $file }
    }
    pack $f_io.btn_dat $f_io.btn_csv -side left -padx 5

    # --- 3D Panel ---
    set f_3d [ttk::labelframe $main.p3d -text " 2. 3D Visualization & Top Plot Settings " -padding "10 5"]
    pack $f_3d -fill x -pady "0 5"

    ttk::label $f_3d.lbl_sel -text "Selection:"
    ttk::entry $f_3d.ent_sel -textvariable ::xAIDashboard::atom_sel -width 24
    ttk::label $f_3d.lbl_style -text " | Style:"
    ttk::combobox $f_3d.opt_style -textvariable ::xAIDashboard::rep_style -values {"Trace" "Tube" "Cartoon" "NewCartoon" "Licorice" "VDW" "Lines"} -width 10
    ttk::checkbutton $f_3d.chk_all -text "Apply to whole res" -variable ::xAIDashboard::apply_all

    ttk::label $f_3d.lbl_metric -text " | Metric:"
    ttk::combobox $f_3d.opt_metric -textvariable ::xAIDashboard::top_metric -values $::xAIDashboard::available_metrics -width 12

    ttk::button $f_3d.btn_color -text "Apply Colors to 3D" -style "Accent.TButton" -command ::xAIDashboard::apply_colors_to_vmd

    # Tutto raggruppato in un'unica riga orizzontale
    pack $f_3d.lbl_sel $f_3d.ent_sel $f_3d.lbl_style $f_3d.opt_style $f_3d.chk_all $f_3d.lbl_metric $f_3d.opt_metric -side left -padx 2
    pack $f_3d.btn_color -side right -padx 5

    # --- Tracks & Zoom Panel ---
    set f_chk [ttk::labelframe $main.chk -text " Overlapping Windows & Zoom " -padding "10 5"]
    pack $f_chk -fill x -pady "0 5"

    ttk::frame $f_chk.container
    pack $f_chk.container -side left

    ttk::frame $f_chk.zcontainer
    pack $f_chk.zcontainer -side right
    ttk::label $f_chk.zcontainer.lbl_z -text "Zoom Window (0=All): \u00b1"
    ttk::entry $f_chk.zcontainer.ent_z -textvariable ::xAIDashboard::zoom_window -width 5
    ttk::label $f_chk.zcontainer.lbl_zf -text "frames"
    pack $f_chk.zcontainer.lbl_z $f_chk.zcontainer.ent_z $f_chk.zcontainer.lbl_zf -side left -padx 2

    # --- Grafici ---
    set f_plt [ttk::labelframe $main.plt -text " Saliency, Metrics & Spatial Analysis " -padding "10 10"]
    pack $f_plt -fill both -expand 1

    canvas $f_plt.c_conf -bg "#FFFFFF" -height 120 -highlightthickness 0
    canvas $f_plt.c_sal -bg "#FFFFFF" -height 160 -highlightthickness 0
    canvas $f_plt.c_space -bg "#FFFFFF" -height 130 -highlightthickness 0

    pack $f_plt.c_conf -fill x -pady "0 5"
    pack $f_plt.c_sal -fill x -pady "0 5"
    pack $f_plt.c_space -fill x

    set ::xAIDashboard::c_conf $f_plt.c_conf
    set ::xAIDashboard::c_sal $f_plt.c_sal
    set ::xAIDashboard::c_space $f_plt.c_space

    # Bindings per l'Hover Dinamico su tutti e 3 i grafici
    bind $f_plt.c_conf <Motion> {::xAIDashboard::on_conf_hover %x %y}
    bind $f_plt.c_conf <Leave> {::xAIDashboard::clear_hover c_conf}

    bind $f_plt.c_sal <Motion> {::xAIDashboard::on_sal_hover %x %y}
    bind $f_plt.c_sal <Leave> {::xAIDashboard::clear_hover c_sal}

    bind $f_plt.c_space <Motion> {::xAIDashboard::on_space_hover %x %y}
    bind $f_plt.c_space <Leave> {::xAIDashboard::clear_hover c_space}

    # Trigger per aggiornare dinamicamente il grafico allo zoom o al cambio metrica
    catch {trace remove variable ::xAIDashboard::zoom_window write ::xAIDashboard::draw_plots_wrapper}
    trace add variable ::xAIDashboard::zoom_window write ::xAIDashboard::draw_plots_wrapper

    catch {trace remove variable ::xAIDashboard::top_metric write ::xAIDashboard::draw_plots_wrapper}
    trace add variable ::xAIDashboard::top_metric write ::xAIDashboard::draw_plots_wrapper
}

proc ::xAIDashboard::draw_plots_wrapper {args} {
    ::xAIDashboard::draw_plots
}

proc ::xAIDashboard::build_dynamic_checkboxes {} {
    set container .xaigui.main.chk.container
    foreach child [winfo children $container] { destroy $child }

    set num_cols [llength $::xAIDashboard::pos_cols]
    if {$num_cols == 0} return

    for {set i 0} {$i < $num_cols} {incr i} {
        set color [lindex $::xAIDashboard::pos_colors $i]
        set f $container.f$i
        ttk::frame $f
        pack $f -side left -padx 5

        canvas $f.c -width 12 -height 12 -bg "#E0E0E0" -highlightthickness 0
        $f.c create oval 2 2 10 10 -fill $color -outline ""
        pack $f.c -side left -pady 2

        ttk::checkbutton $f.cb -text "Track $i" -variable ::xAIDashboard::show_pos($i) -command ::xAIDashboard::draw_plots
        pack $f.cb -side left -padx 2
    }
}
