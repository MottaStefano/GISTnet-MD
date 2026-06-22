namespace eval ::xAIDashboard {}

proc ::xAIDashboard::draw_plots {} {
    if {$::xAIDashboard::num_frames == 0} return

    set c_sal $::xAIDashboard::c_sal
    set c_conf $::xAIDashboard::c_conf
    set c_space $::xAIDashboard::c_space

    $c_sal delete all
    $c_conf delete all
    $c_space delete all

    # Forza l'aggiornamento della UI per ottenere le dimensioni REALI del Canvas
    update idletasks

    set w [winfo width $c_sal]
    if {$w < 10} { set w 600 }

    set h_conf [winfo height $c_conf]
    set h_sal [winfo height $c_sal]
    set h_space [winfo height $c_space]

    set pad_l 55
    set pad_r 20
    set pad_t 25
    set pad_b 25

    # 1. Trova dinamicamente i limiti (Min e Max) della metrica selezionata in alto
    set min_metric 9999.0
    set max_metric -9999.0
    for {set f 0} {$f < $::xAIDashboard::num_frames} {incr f} {
        if {[info exists ::xAIDashboard::time_data($f,$::xAIDashboard::top_metric)]} {
            set val $::xAIDashboard::time_data($f,$::xAIDashboard::top_metric)
            if {$val > $max_metric} { set max_metric $val }
            if {$val < $min_metric} { set min_metric $val }
        }
    }
    if {$max_metric < $min_metric} { set min_metric 0.0; set max_metric 1.0 }

    # 2. Impostazione della Baseline e Ricalcolo Simmetrico degli Assi
    set base_v ""
    if {$::xAIDashboard::top_metric eq "Directional"} {
        set base_v 0.0
        set max_abs [expr {max(abs($max_metric), abs($min_metric))}]
        if {$max_abs == 0} { set max_abs 1.0 }
        set min_metric [expr {-$max_abs}]
        set max_metric $max_abs
    } elseif {$::xAIDashboard::top_metric eq "Confidence"} {
        set base_v 0.5
        set max_dist [expr {max(abs($max_metric - 0.5), abs($min_metric - 0.5))}]
        if {$max_dist == 0} { set max_dist 0.5 }
        set min_metric [expr {0.5 - $max_dist}]
        set max_metric [expr {0.5 + $max_dist}]
    } elseif {$max_metric == $min_metric} {
        set max_metric [expr {$min_metric + 1.0}]
    }

    set range_metric [expr {$max_metric - $min_metric}]
    if {$range_metric == 0} { set range_metric 1.0 }

    # --- Assegnazione Assi ---
    ::xAIDashboard::draw_axes $c_conf "Top Metric: $::xAIDashboard::top_metric" $min_metric $max_metric $pad_l $pad_r $pad_t $pad_b 2 $w $h_conf
    ::xAIDashboard::draw_axes $c_sal "Overlapping Windows Saliency" 0.0 $::xAIDashboard::max_sal $pad_l $pad_r $pad_t $pad_b 4 $w $h_sal
    ::xAIDashboard::draw_axes $c_space "Instantaneous Spatial Profile (Per-Residue)" $::xAIDashboard::min_spat $::xAIDashboard::max_spat $pad_l $pad_r $pad_t $pad_b 2 $w $h_space

    # Assi X statici
    ::xAIDashboard::draw_xaxis_static $c_conf 0 [expr {$::xAIDashboard::num_frames - 1}] $pad_l $pad_r $pad_b $w $h_conf 0
    if {$::xAIDashboard::num_res > 0} {
        ::xAIDashboard::draw_xaxis_static $c_space 0 [expr {$::xAIDashboard::num_res - 1}] $pad_l $pad_r $pad_b $w $h_space 1
    }

    # =========================================================================
    # --- PLOT TOP METRIC (Con Split Dinamico Verde/Rosso e Ultra-Fast Downsampling) ---
    # =========================================================================
    set draw_w [expr {$w - $pad_l - $pad_r}]
    set draw_h_conf [expr {$h_conf - $pad_t - $pad_b}]
    set nframes [expr {$::xAIDashboard::num_frames > 1 ? ($::xAIDashboard::num_frames - 1) : 1}]

    # Inviluppo "al volo" per salvare memoria
    set dec_pts {}
    set last_int_x -100
    set b_min_v 99999.0
    set b_max_v -99999.0

    for {set f 0} {$f < $::xAIDashboard::num_frames} {incr f} {
        set x [expr {$pad_l + ($f * 1.0 / $nframes) * $draw_w}]
        set val 0.0
        if {[info exists ::xAIDashboard::time_data($f,$::xAIDashboard::top_metric)]} {
            set val $::xAIDashboard::time_data($f,$::xAIDashboard::top_metric)
        }

        set int_x [expr {int($x)}]
        if {$int_x != $last_int_x} {
            if {$last_int_x != -100} {
                set y_min [expr {($h_conf - $pad_b) - (($b_min_v - $min_metric) / $range_metric) * $draw_h_conf}]
                lappend dec_pts [list $last_int_x $b_min_v $y_min]
                if {$b_max_v != $b_min_v} {
                    set y_max [expr {($h_conf - $pad_b) - (($b_max_v - $min_metric) / $range_metric) * $draw_h_conf}]
                    lappend dec_pts [list $last_int_x $b_max_v $y_max]
                }
            }
            set last_int_x $int_x
            set b_min_v $val
            set b_max_v $val
        } else {
            if {$val < $b_min_v} { set b_min_v $val }
            if {$val > $b_max_v} { set b_max_v $val }
        }
    }
    if {$last_int_x != -100} {
        set y_min [expr {($h_conf - $pad_b) - (($b_min_v - $min_metric) / $range_metric) * $draw_h_conf}]
        lappend dec_pts [list $last_int_x $b_min_v $y_min]
        if {$b_max_v != $b_min_v} {
            set y_max [expr {($h_conf - $pad_b) - (($b_max_v - $min_metric) / $range_metric) * $draw_h_conf}]
            lappend dec_pts [list $last_int_x $b_max_v $y_max]
        }
    }

    set points {}
    set prev_val ""
    set prev_x ""
    set prev_y ""

    # Calcolo esatto delle intersezioni sulla sequenza ottimizzata
    foreach pt $dec_pts {
        set x [lindex $pt 0]
        set val [lindex $pt 1]
        set y [lindex $pt 2]

        if {$base_v ne "" && $prev_val ne ""} {
            if {($prev_val >= $base_v && $val < $base_v) || ($prev_val < $base_v && $val >= $base_v)} {
                set t [expr {($base_v - $prev_val) / ($val - $prev_val)}]
                set int_x [expr {$prev_x + $t * ($x - $prev_x)}]
                set int_y [expr {($h_conf - $pad_b) - (($base_v - $min_metric) / $range_metric) * $draw_h_conf}]
                lappend points [list $int_x $base_v $int_y]
            }
        }
        lappend points [list $x $val $y]
        set prev_val $val; set prev_x $x; set prev_y $y
    }

    if {$base_v ne ""} {
        # Logica per tracciare poligoni spezzati (Verdi / Rossi)
        set lines_green {}
        set lines_red {}
        set cur_line {}
        set cur_mode ""

        foreach pt $points {
            set x [lindex $pt 0]
            set v [lindex $pt 1]
            set y [lindex $pt 2]

            if {abs($v - $base_v) < 1e-6} {
                set pt_mode "boundary"
            } elseif {$v > $base_v} {
                set pt_mode "green"
            } else {
                set pt_mode "red"
            }

            if {$cur_mode eq ""} {
                if {$pt_mode ne "boundary"} { set cur_mode $pt_mode }
                lappend cur_line $x $y
            } else {
                if {$pt_mode eq "boundary"} {
                    lappend cur_line $x $y
                    if {$cur_mode eq "green"} { lappend lines_green $cur_line }
                    if {$cur_mode eq "red"} { lappend lines_red $cur_line }
                    set cur_line [list $x $y]
                    set cur_mode ""
                } elseif {$pt_mode eq $cur_mode} {
                    lappend cur_line $x $y
                } else {
                    if {$cur_mode eq "green"} { lappend lines_green $cur_line }
                    if {$cur_mode eq "red"} { lappend lines_red $cur_line }
                    set cur_line [list $x $y]
                    set cur_mode $pt_mode
                }
            }
        }
        if {[llength $cur_line] >= 4} {
            if {$cur_mode eq "green"} { lappend lines_green $cur_line }
            if {$cur_mode eq "red"} { lappend lines_red $cur_line }
        }

        set base_y [expr {($h_conf - $pad_b) - (($base_v - $min_metric) / $range_metric) * $draw_h_conf}]
        $c_conf create line $pad_l $base_y [expr {$w - $pad_r}] $base_y -fill "#888888" -dash "." -width 1 -tags "data"

        foreach line $lines_green {
            if {[llength $line] < 4} continue
            set poly [list [lindex $line 0] $base_y]
            lappend poly {*}$line
            lappend poly [lindex $line end-1] $base_y
            $c_conf create polygon {*}$poly -fill "#D5F5E3" -outline "" -tags "data"
            $c_conf create line {*}$line -fill "#2ECC71" -width 2 -joinstyle round -tags "data"
        }
        foreach line $lines_red {
            if {[llength $line] < 4} continue
            set poly [list [lindex $line 0] $base_y]
            lappend poly {*}$line
            lappend poly [lindex $line end-1] $base_y
            $c_conf create polygon {*}$poly -fill "#FADBD8" -outline "" -tags "data"
            $c_conf create line {*}$line -fill "#E74C3C" -width 2 -joinstyle round -tags "data"
        }
    } else {
        # Metriche normali senza Split (es. Anomaly_Score)
        set line_coords {}
        set poly_coords [list $pad_l [expr {$h_conf - $pad_b}]]
        foreach pt $points {
            set x [lindex $pt 0]
            set y [lindex $pt 2]
            lappend line_coords $x $y
            lappend poly_coords $x $y
        }
        lappend poly_coords [expr {$pad_l + $draw_w}] [expr {$h_conf - $pad_b}]
        $c_conf create polygon {*}$poly_coords -fill "#D0E8FF" -outline "" -tags "data"
        $c_conf create line {*}$line_coords -fill "#0077CC" -width 2 -joinstyle round -tags "data"
    }

    ::xAIDashboard::update_cursor [::vmd_trajectory_read]
}

proc ::xAIDashboard::draw_axes {c title min_val max_val pad_l pad_r pad_t pad_b steps w h} {
    set draw_h [expr {$h - $pad_t - $pad_b}]

    $c create line $pad_l $pad_t $pad_l [expr {$h - $pad_b}] -fill "#000000" -width 1
    $c create line $pad_l [expr {$h - $pad_b}] [expr {$w - $pad_r}] [expr {$h - $pad_b}] -fill "#000000" -width 1
    $c create text $pad_l [expr {$pad_t - 15}] -anchor nw -text $title -fill "#000000" -font "Helvetica 9 bold"

    for {set i 0} {$i <= $steps} {incr i} {
        set frac [expr {double($i) / $steps}]
        set y [expr {($h - $pad_b) - $frac * $draw_h}]
        set val [expr {$min_val + $frac * ($max_val - $min_val)}]
        set label_text [format "%g" $val]

        if {$i > 0} { $c create line $pad_l $y [expr {$w - $pad_r}] $y -fill "#E0E0E0" -dash "." }
        $c create line [expr {$pad_l - 4}] $y $pad_l $y -fill "#000000"
        $c create text [expr {$pad_l - 8}] $y -anchor e -text $label_text -fill "#333333" -font "Helvetica 8"
    }
}

proc ::xAIDashboard::draw_xaxis_static {c min_val max_val pad_l pad_r pad_b w h {is_spatial 0}} {
    set draw_w [expr {$w - $pad_l - $pad_r}]
    set steps 5

    if {$is_spatial && [info exists ::xAIDashboard::res_ids] && [llength $::xAIDashboard::res_ids] > 0} {
        set min_val [lindex $::xAIDashboard::res_ids 0]
        set max_val [lindex $::xAIDashboard::res_ids end]
    }

    for {set i 0} {$i <= $steps} {incr i} {
        set frac [expr {double($i) / $steps}]
        set x [expr {$pad_l + $frac * $draw_w}]
        set val [expr {int($min_val + $frac * ($max_val - $min_val))}]

        $c create line $x [expr {$h - $pad_b}] $x [expr {$h - $pad_b + 4}] -fill "#000000"
        $c create text $x [expr {$h - $pad_b + 8}] -anchor n -text $val -fill "#333333" -font "Helvetica 8" -tags "x_axis_labels"
    }
}

proc ::xAIDashboard::update_cursor {frame} {
    if {$::xAIDashboard::num_frames == 0} return
    set c_sal $::xAIDashboard::c_sal
    set c_conf $::xAIDashboard::c_conf
    set c_space $::xAIDashboard::c_space

    set pad_l 55; set pad_r 20; set pad_t 25; set pad_b 25

    set w [winfo width $c_sal]
    if {$w < 10} { set w 600 }

    set h_sal [winfo height $c_sal]
    set h_conf [winfo height $c_conf]
    set h_space [winfo height $c_space]

    set draw_w [expr {$w - $pad_l - $pad_r}]
    set draw_h_sal [expr {$h_sal - $pad_t - $pad_b}]

    if {$frame >= $::xAIDashboard::num_frames} { set frame [expr {$::xAIDashboard::num_frames - 1}] }
    if {$frame < 0} { set frame 0 }

    # 1. Update Top Plot Cursor
    $c_conf delete cursor
    set nframes_global [expr {$::xAIDashboard::num_frames > 1 ? ($::xAIDashboard::num_frames - 1) : 1}]
    set x_global [expr {$pad_l + ($frame * 1.0 / $nframes_global) * $draw_w}]
    $c_conf create line $x_global $pad_t $x_global [expr {$h_conf - $pad_b}] -fill "#000000" -width 2 -tags "cursor"

    # 2. Update Saliency (Gestione Zoom, Ultra-Fast Downsampling)
    $c_sal delete data
    $c_sal delete "x_axis_labels"
    $c_sal delete cursor

    set z 0
    catch { set z [expr {int($::xAIDashboard::zoom_window)}] }

    set start_f 0
    set end_f [expr {$::xAIDashboard::num_frames - 1}]

    if {$z > 0} {
        set start_f [expr {$frame - $z}]
        set end_f [expr {$frame + $z}]
    }
    set span [expr {$end_f - $start_f}]
    if {$span <= 0} { set span 1 }

    set num_lines [llength $::xAIDashboard::pos_cols]
    for {set i 0} {$i < $num_lines} {incr i} {
        if {$::xAIDashboard::show_pos($i) == 0} continue
        set color [lindex $::xAIDashboard::pos_colors $i]

        set line_coords {}
        set last_int_x -100
        set b_min_y 99999.0
        set b_max_y -99999.0

        for {set f $start_f} {$f <= $end_f} {incr f} {
            if {$f < 0 || $f >= $::xAIDashboard::num_frames} {
                set val "NaN"
            } else {
                set val [lindex $::xAIDashboard::time_data($f,sals) $i]
            }

            if {$val eq "NaN"} {
                if {$last_int_x != -100} {
                    lappend line_coords $last_int_x $b_min_y
                    if {$b_max_y != $b_min_y} { lappend line_coords $last_int_x $b_max_y }
                    if {[llength $line_coords] >= 4} { $c_sal create line {*}$line_coords -fill $color -width 2 -joinstyle round -tags "data" }
                }
                set line_coords {}
                set last_int_x -100
            } else {
                set x [expr {$pad_l + (($f - $start_f) * 1.0 / $span) * $draw_w}]
                set y [expr {($h_sal - $pad_b) - ($val / $::xAIDashboard::max_sal) * $draw_h_sal}]
                set int_x [expr {int($x)}]

                if {$int_x != $last_int_x} {
                    if {$last_int_x != -100} {
                        lappend line_coords $last_int_x $b_min_y
                        if {$b_max_y != $b_min_y} { lappend line_coords $last_int_x $b_max_y }
                    }
                    set last_int_x $int_x
                    set b_min_y $y
                    set b_max_y $y
                } else {
                    if {$y < $b_min_y} { set b_min_y $y }
                    if {$y > $b_max_y} { set b_max_y $y }
                }
            }
        }

        if {$last_int_x != -100} {
            lappend line_coords $last_int_x $b_min_y
            if {$b_max_y != $b_min_y} { lappend line_coords $last_int_x $b_max_y }
            if {[llength $line_coords] >= 4} { $c_sal create line {*}$line_coords -fill $color -width 2 -joinstyle round -tags "data" }
        }
    }

    ::xAIDashboard::draw_xaxis_static $c_sal $start_f $end_f $pad_l $pad_r $pad_b $w $h_sal 0

    set x_sal [expr {$pad_l + (($frame - $start_f) * 1.0 / $span) * $draw_w}]
    $c_sal create line $x_sal $pad_t $x_sal [expr {$h_sal - $pad_b}] -fill "#000000" -width 2 -tags "cursor"

    # 3. Update Spatial Profile (Supporto Valori Negativi, Gaps e Ultra-Fast Downsampling)
    $c_space delete data
    if {[info exists ::xAIDashboard::spat_data($frame)] && $::xAIDashboard::num_res > 1} {
        set vals $::xAIDashboard::spat_data($frame)
        set draw_h_space [expr {$h_space - $pad_t - $pad_b}]

        set range_spat [expr {$::xAIDashboard::max_spat - $::xAIDashboard::min_spat}]
        if {$range_spat == 0} { set range_spat 1.0 }

        set s_coords {}
        set last_int_x -100
        set b_min_y 99999.0
        set b_max_y -99999.0

        set has_res_ids [info exists ::xAIDashboard::res_ids]
        set span_res [expr {$::xAIDashboard::num_res - 1}]

        if {$has_res_ids && [llength $::xAIDashboard::res_ids] > 1} {
            set span_res [expr {[lindex $::xAIDashboard::res_ids end] - [lindex $::xAIDashboard::res_ids 0]}]
            if {$span_res <= 0} { set span_res 1 }
        }

        for {set i 0} {$i < [llength $vals]} {incr i} {
            if {$has_res_ids} {
                set curr_res [lindex $::xAIDashboard::res_ids $i]
                set min_res [lindex $::xAIDashboard::res_ids 0]
                set x [expr {$pad_l + (($curr_res - $min_res) * 1.0 / $span_res) * $draw_w}]
            } else {
                set x [expr {$pad_l + ($i * 1.0 / ($::xAIDashboard::num_res - 1)) * $draw_w}]
            }

            set is_gap 0
            if {$has_res_ids && $i > 0} {
                set prev_res [lindex $::xAIDashboard::res_ids [expr {$i - 1}]]
                if {($curr_res - $prev_res) > 1} {
                    set is_gap 1
                }
            }

            if {$is_gap} {
                if {$last_int_x != -100} {
                    lappend s_coords $last_int_x $b_min_y
                    if {$b_max_y != $b_min_y} { lappend s_coords $last_int_x $b_max_y }
                    if {[llength $s_coords] >= 4} { $c_space create line {*}$s_coords -fill "#FF9800" -width 2 -joinstyle round -tags "data" }
                }
                set s_coords {}
                set last_int_x -100
            }

            set val [lindex $vals $i]
            if {$val eq ""} { set val 0.0 }

            set y [expr {($h_space - $pad_b) - (($val - $::xAIDashboard::min_spat) / $range_spat) * $draw_h_space}]
            set int_x [expr {int($x)}]

            if {$int_x != $last_int_x} {
                if {$last_int_x != -100} {
                    lappend s_coords $last_int_x $b_min_y
                    if {$b_max_y != $b_min_y} { lappend s_coords $last_int_x $b_max_y }
                }
                set last_int_x $int_x
                set b_min_y $y
                set b_max_y $y
            } else {
                if {$y < $b_min_y} { set b_min_y $y }
                if {$y > $b_max_y} { set b_max_y $y }
            }
        }

        if {$last_int_x != -100} {
            lappend s_coords $last_int_x $b_min_y
            if {$b_max_y != $b_min_y} { lappend s_coords $last_int_x $b_max_y }
            if {[llength $s_coords] >= 4} { $c_space create line {*}$s_coords -fill "#FF9800" -width 2 -joinstyle round -tags "data" }
        }
    }
}

# =============================================================================
# GESTIONE HOVER TOOLTIP (SU TUTTI I GRAFICI)
# =============================================================================
proc ::xAIDashboard::draw_hover_tooltip {c mouse_x mouse_y exact_x text} {
    set w [winfo width $c]
    set h [winfo height $c]

    $c create line $exact_x 25 $exact_x [expr {$h - 25}] -fill "#777777" -dash "." -tags "hover_info"

    set tx [expr {$exact_x + 8}]
    set ty [expr {$mouse_y - 15}]
    set anchor "w"

    if {$tx > $w - 100} { set tx [expr {$exact_x - 8}]; set anchor "e" }
    if {$ty < 35} { set ty 35 }

    set text_id [$c create text $tx $ty -text $text -anchor $anchor -font "Helvetica 9 bold" -fill "#333333" -tags "hover_info"]
    set bbox [$c bbox $text_id]
    if {[llength $bbox] == 4} {
        set pad 4
        set bg_id [$c create rectangle [expr {[lindex $bbox 0]-$pad}] [expr {[lindex $bbox 1]-$pad}] [expr {[lindex $bbox 2]+$pad}] [expr {[lindex $bbox 3]+$pad}] -fill "#FFFFDD" -outline "#AAAAAA" -tags "hover_info_bg"]
        $c raise $text_id
    }
}

proc ::xAIDashboard::clear_hover {canvas_name} {
    set c [set ::xAIDashboard::$canvas_name]
    $c delete hover_info
    $c delete hover_info_bg
}

proc ::xAIDashboard::on_conf_hover {x y} {
    set c $::xAIDashboard::c_conf
    ::xAIDashboard::clear_hover c_conf
    if {$::xAIDashboard::num_frames == 0} return

    set pad_l 55; set pad_r 20; set w [winfo width $c]
    set draw_w [expr {$w - $pad_l - $pad_r}]
    if {$x < $pad_l || $x > [expr {$w - $pad_r}]} return

    set frac [expr {($x - $pad_l) * 1.0 / $draw_w}]
    set frame [expr {int(round($frac * ($::xAIDashboard::num_frames - 1)))}]
    if {$frame < 0} { set frame 0 }
    if {$frame >= $::xAIDashboard::num_frames} { set frame [expr {$::xAIDashboard::num_frames - 1}] }

    set val 0.0
    if {[info exists ::xAIDashboard::time_data($frame,$::xAIDashboard::top_metric)]} {
        set val $::xAIDashboard::time_data($frame,$::xAIDashboard::top_metric)
    }

    set exact_x [expr {$pad_l + ($frame * 1.0 / ($::xAIDashboard::num_frames > 1 ? ($::xAIDashboard::num_frames - 1) : 1)) * $draw_w}]
    set text "Frame: $frame\n$::xAIDashboard::top_metric: [format "%.3f" $val]"

    ::xAIDashboard::draw_hover_tooltip $c $x $y $exact_x $text
}

proc ::xAIDashboard::on_sal_hover {x y} {
    set c $::xAIDashboard::c_sal
    ::xAIDashboard::clear_hover c_sal
    if {$::xAIDashboard::num_frames == 0} return

    set pad_l 55; set pad_r 20; set w [winfo width $c]
    set draw_w [expr {$w - $pad_l - $pad_r}]
    if {$x < $pad_l || $x > [expr {$w - $pad_r}]} return

    set z 0
    catch { set z [expr {int($::xAIDashboard::zoom_window)}] }

    set current_frame [::vmd_trajectory_read]
    set start_f 0
    set end_f [expr {$::xAIDashboard::num_frames - 1}]

    if {$z > 0} {
        set start_f [expr {$current_frame - $z}]
        set end_f [expr {$current_frame + $z}]
    }
    set span [expr {$end_f - $start_f}]
    if {$span <= 0} { set span 1 }

    set frac [expr {($x - $pad_l) * 1.0 / $draw_w}]
    set frame [expr {int(round($start_f + $frac * $span))}]
    if {$frame < 0 || $frame >= $::xAIDashboard::num_frames} return

    set max_val 0.0
    set num_lines [llength $::xAIDashboard::pos_cols]
    for {set i 0} {$i < $num_lines} {incr i} {
        if {$::xAIDashboard::show_pos($i) == 0} continue
        if {[info exists ::xAIDashboard::time_data($frame,sals)]} {
            set v [lindex $::xAIDashboard::time_data($frame,sals) $i]
            if {$v ne "NaN" && $v > $max_val} { set max_val $v }
        }
    }

    set exact_x [expr {$pad_l + (($frame - $start_f) * 1.0 / $span) * $draw_w}]
    set text "Frame: $frame\nMax Saliency: [format "%.3f" $max_val]"

    ::xAIDashboard::draw_hover_tooltip $c $x $y $exact_x $text
}

proc ::xAIDashboard::on_space_hover {x y} {
    set c $::xAIDashboard::c_space
    ::xAIDashboard::clear_hover c_space

    if {$::xAIDashboard::num_frames == 0 || $::xAIDashboard::num_res < 1} return

    set pad_l 55
    set pad_r 20
    set w [winfo width $c]
    set draw_w [expr {$w - $pad_l - $pad_r}]

    if {$x < $pad_l || $x > [expr {$w - $pad_r}]} return

    set frac [expr {($x - $pad_l) * 1.0 / $draw_w}]
    set found_res ""
    set found_name ""
    set found_idx -1
    set exact_x $x

    if {[info exists ::xAIDashboard::res_ids] && [llength $::xAIDashboard::res_ids] > 1} {
        set min_res [lindex $::xAIDashboard::res_ids 0]
        set max_res [lindex $::xAIDashboard::res_ids end]
        set span_res [expr {$max_res - $min_res}]
        set target_res [expr {$min_res + $frac * $span_res}]

        set min_dist 9999
        for {set i 0} {$i < [llength $::xAIDashboard::res_ids]} {incr i} {
            set r [lindex $::xAIDashboard::res_ids $i]
            set dist [expr {abs($r - $target_res)}]
            if {$dist < $min_dist} {
                set min_dist $dist
                set found_res $r
                if {[info exists ::xAIDashboard::res_names]} {
                    set found_name [lindex $::xAIDashboard::res_names $i]
                }
                set found_idx $i
                set exact_x [expr {$pad_l + (($r - $min_res) * 1.0 / $span_res) * $draw_w}]
            }
        }
    } else {
        set target_idx [expr {int(round($frac * ($::xAIDashboard::num_res - 1)))}]
        if {$target_idx >= 0 && $target_idx < $::xAIDashboard::num_res} {
            set found_idx $target_idx
            set found_res $target_idx
            set exact_x [expr {$pad_l + ($target_idx * 1.0 / ($::xAIDashboard::num_res - 1)) * $draw_w}]
        }
    }

    if {$found_idx != -1} {
        set f [::vmd_trajectory_read]
        if {$f >= $::xAIDashboard::num_frames} { set f [expr {$::xAIDashboard::num_frames - 1}] }
        if {$f < 0} { set f 0 }

        set val 0.0
        if {[info exists ::xAIDashboard::spat_data($f)]} {
            set val_raw [lindex $::xAIDashboard::spat_data($f) $found_idx]
            if {$val_raw ne ""} { set val $val_raw }
        }

        set label_res "Residuo: $found_res"
        if {$found_name ne ""} { set label_res "$found_name$found_res" }
        set text "$label_res\nValore: [format "%.3f" $val]"

        ::xAIDashboard::draw_hover_tooltip $c $x $y $exact_x $text
    }
}
