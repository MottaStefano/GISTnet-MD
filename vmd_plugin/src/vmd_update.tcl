namespace eval ::xAIDashboard {}

proc ::xAIDashboard::setup_vmd_trace {} {
    set ::xAIDashboard::active_mol [molinfo top]
    if {$::xAIDashboard::active_mol == -1} { puts "Nessuna molecola in VMD!"; return }

    if {!$::xAIDashboard::trace_active} {
        trace add variable ::vmd_frame($::xAIDashboard::active_mol) write ::xAIDashboard::vmd_time_callback
        set ::xAIDashboard::trace_active 1
        puts "VMD Time Trace collegata."
    }
    ::xAIDashboard::vmd_time_callback foo foo foo
}

proc ::xAIDashboard::vmd_time_callback {name element op} {
    set f [::vmd_trajectory_read]
    ::xAIDashboard::update_cursor $f
}

proc ::vmd_trajectory_read {} {
    if {$::xAIDashboard::active_mol == -1} { return 0 }
    return [molinfo $::xAIDashboard::active_mol get frame]
}

proc ::xAIDashboard::apply_colors_to_vmd {} {
    variable active_mol
    variable spat_data
    variable num_res
    variable atom_sel
    variable apply_all
    variable rep_style

    set active_mol [molinfo top]
    if {$active_mol == -1 || ![info exists spat_data(0)]} {
        tk_messageBox -type ok -icon error -message "Carica prima una molecola in VMD e i file Dati."
        return
    }

    set vmd_frames [molinfo $active_mol get numframes]

    # 1. Troviamo i residui e i PDB resid all'interno della selezione
    set ca_sel [atomselect $active_mol $atom_sel]

    set u_residues {}
    set u_resids {}
    set u_resnames {}
    # Estrae in parallelo indice (residue), ID PDB (resid) e nome (resname)
    foreach r [$ca_sel get residue] rid [$ca_sel get resid] rname [$ca_sel get resname] {
        if {[lsearch -exact $u_residues $r] == -1} {
            lappend u_residues $r
            lappend u_resids $rid
            lappend u_resnames $rname
        }
    }

    set n_unique [llength $u_residues]
    if {$n_unique != $num_res} {
        puts "ATTENZIONE: Trovati $n_unique residui unici nella selezione, ma i dati spaziali hanno $num_res nodi!"
    }

    # Salviamo i residui PDB reali e i nomi per l'asse x e hover
    set ::xAIDashboard::res_ids $u_resids
    set ::xAIDashboard::res_names $u_resnames

    # Creiamo una mappa: ID Residuo (indice VMD) -> Indice del Nodo Dati (0, 1, ..., N-1)
    array set res2node {}
    set node_idx 0
    foreach r $u_residues {
        set res2node($r) $node_idx
        incr node_idx
    }

    # 2. Definiamo la selezione bersaglio su cui applicare i colori
    if {$apply_all} {
        set target_sel [atomselect $active_mol "same residue as ($atom_sel)"]
    } else {
        set target_sel [atomselect $active_mol $atom_sel]
    }

    # 3. Mappiamo ogni singolo atomo del bersaglio al suo indice nodo
    set target_res_list [$target_sel get residue]
    set atom2node {}
    foreach r $target_res_list {
        if {[info exists res2node($r)]} {
            lappend atom2node $res2node($r)
        } else {
            lappend atom2node 0
        }
    }

    # 4. Applichiamo i dati frame per frame
    for {set i 0} {$i < $vmd_frames} {incr i} {
        if {[info exists spat_data($i)]} {
            set val_list $spat_data($i)
            $target_sel frame $i

            # Sanitizzazione e mappatura per atomo finale
            set safe_vals {}
            foreach idx $atom2node {
                set val [lindex $val_list $idx]
                if {![string is double -strict $val]} { set val 0.0 }
                lappend safe_vals $val
            }
            $target_sel set user $safe_vals
        }
    }

    $ca_sel delete
    $target_sel delete

    # 5. Aggiornamento Stili VMD
    mol delrep 0 $active_mol
    mol representation $rep_style
    mol color User
    mol selection "protein"
    mol addrep $active_mol

    color scale method BWR

    # Scala i colori di default tra -5 e +5
    mol scaleminmax $active_mol 0 -5.0 5.0

    set cur_frame [molinfo $active_mol get frame]
    molinfo $active_mol set frame $cur_frame
    puts "Colori applicati con stile $rep_style per $n_unique residui!"

    # Aggiorna il Canvas per iniettare i veri PDB Resid ID appena estratti
    catch { ::xAIDashboard::draw_plots }
}
