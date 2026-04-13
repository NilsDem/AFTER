// inlets & outlets
inlets = 2
outlets = 2
// Global current model
var cur_prior = 0;
var cur_encoder = 0;
var cur_decoder = 0;
var timbre_recv = 0;
var timbre_unp = 0;
var cur_z = 0;
// Instance prefix
var prefix_z = "";
var prefix_p = "";
var prefix_s = "";
// Object creation variable
var p = this.patcher

var cur_connect = []

// Set prefix
function prefix(val_z, val_p, val_s)
{
	prefix_z = val_z
	prefix_p = val_p
	prefix_s = val_s
}

function set_knobs_enabled(state) {
    // state = 1 → enable
    // state = 0 → disable

    var ignore = state ? 0 : 1;

    // ----- bias0..bias7 -----
    for (var i = 0; i < 8; i++) {
        var b = this.patcher.getnamed("bias" + i);
        if (b) {
            b.message("active", state);
            b.message("ignoreclick", ignore);
        }
        var s = this.patcher.getnamed("scale" + i);
        if (s) {
            s.message("active", state);
            s.message("ignoreclick", ignore);
        }
    }

    // ----- globals -----
    var bg = this.patcher.getnamed("bias_global");
    if (bg) {
        bg.message("active", state);
        bg.message("ignoreclick", ignore);
    }

    var sg = this.patcher.getnamed("scale_global");
    if (sg) {
        sg.message("active", state);
        sg.message("ignoreclick", ignore);
    }

    post("Knobs set to state = " + state + "\n");
}


function set_warning_message(msg) {
    var w = this.patcher.getnamed("warning_msg");
    if (w) {
        w.message("text", msg);
    }

	var panel = this.patcher.getnamed("panel_latents");
    if (panel) {
        panel.message("hidden", 0);   // show
    }


}


function clear_warning_message() {
    var w = this.patcher.getnamed("warning_msg");
    if (w) {
        w.message("text", "");
    }
	var panel = this.patcher.getnamed("panel_latents");
    if (panel) {
        panel.message("hidden", 1);   // show
    }

}


// Create a given model
function create(model_name, buffer_size, prior) 

{
	post("Received:", model_name, buffer_size, prior, "\n");

	
	if (String(model_name).indexOf("off") !== -1)//(model_name == "off")
	{
		_delete_current()
		return
	}
	else 
	{
		_delete_current()	
	}
	// Replace current model
	if (cur_z != 0)
		_delete_current()

	// Create nn~s
	
	//var structure_encoder = p.newdefault(100, 100, "nn~", model_name, "structure", buffer_size);
	//structure_encoder.rect = [100, 100, 500, 25]
	
	
	
	var max_decoder = p.newdefault(100, 350, "nn~", model_name, "diffuse_timbre_modulate", buffer_size);
	max_decoder.rect = [100, 350, 850, 25]
	
	var main_plot = p.getnamed("main_plot");
	main_plot.setattr("bkgndpict", model_name + ".png")
		
	// ----------- IF / ELSE TEST -----------
if (max_decoder && max_decoder.maxclass === "nn~") {
    post("[OK] External nn~ loaded successfully\n");
	set_knobs_enabled(1);
	clear_warning_message();
    // put your OK-path code here

	var max_codec_decoder = p.newdefault(100, 600, "nn~", model_name, "decode", buffer_size);
	max_codec_decoder.rect = [100, 600, 850, 25]
	

	
	//cur_connect.push(structure_encoder)
	cur_connect.push(max_decoder)
	cur_connect.push(max_codec_decoder)
	
	
	// Number of latents
	
	var n_latents_timbre = 6
	var ninlets = max_decoder.getboxattr('numinlets')

	var n_latents_structure = (max_decoder.getboxattr('numinlets') - n_latents_timbre - 1)/2
	
	post("Hlatent structure",n_latents_structure);
	post("Hlatent inlents",ninlets);

	
	var n_ae_latents  = max_decoder.getboxattr('numoutlets')

	// Gather all 
	cur_z = []

	
	// Make major in / out connections
	p.connect(p.getnamed("plug_in"), 0, max_decoder, 0)
	p.connect(p.getnamed("model_props"), 0, max_decoder, 0)
	p.connect(p.getnamed("model_props"), 0, max_codec_decoder, 0)
	//p.connect(p.getnamed("model_props"), 0, structure_encoder, 0)
	p.connect(p.getnamed("msg_in"), 0, max_decoder, 0)
	p.connect(max_codec_decoder, 0, p.getnamed("rave_out"), 0)
	
	// Create prior

	// ### DIFFUSION MODEL ### 
	for (var z = 0; z < n_latents_structure; z+=1)
	{
		// cur_connect = []
		

		if (z < 8)
		{	
			
			var bias_mod = p.newdefault(210 + (100 * z),  190, "r~", ""+prefix_s+""+z+"_s");
			bias_mod.rect = [210 + (100 * z), 190, 240 + (100 * z), 220]
			
			var scale_mod = p.newdefault(250 + (100 * z), 190, "r~", ""+prefix_s+""+z+"_b");
			scale_mod.rect = [250 + (100 * z), 190, 280 + (100 * z), 220]
			
			p.connect(bias_mod, 0, max_decoder, 1+ n_latents_timbre+ 2*z)
			p.connect(scale_mod, 0, max_decoder, 1+ n_latents_timbre+ 2*z + 1)
			cur_connect.push(bias_mod)
			cur_connect.push(scale_mod)
			cur_z.push(cur_connect)
			continue
		}
		
		var sig_scale = p.newdefault(210 + (100 * z), 190, "sig~", 1.);
		//sig_scale.message(1.)
		sig_scale.rect = [210 + (100 * z), 190, 240 + (100 * z), 220]
		
		
		var sig_bias = p.newdefault(250 + (100 * z), 190, "sig~", 0.);
		//sig_bias.message(0..)
		sig_bias.rect = [250 + (100 * z), 190, 280 + (100 * z), 220]

		p.connect(sig_scale, 0, max_decoder, 1 +n_latents_timbre+ 2*z);
		p.connect(sig_bias, 0, max_decoder, 1 +n_latents_timbre+ 2*z+1);
		cur_connect.push(sig_bias)
		cur_connect.push(sig_scale)
	}




} else {
    post("[ERROR] Failed to load nn~ external\n");
    // put your fallback or alternative logic here
	set_knobs_enabled(0);
	p.remove(max_decoder);
	set_warning_message("V5 model required");

	var max_decoder = p.newdefault(100, 350, "nn~", model_name,"generate_timbre", buffer_size);
	max_decoder.rect = [100, 350, 850, 25]
	
	var n_latents_timbre = max_decoder.getboxattr('numinlets') - 1
	
	
	p.connect(p.getnamed("plug_in"), 0, max_decoder, 0)
	p.connect(p.getnamed("model_props"), 0, max_decoder, 0)
	p.connect(max_decoder, 0, p.getnamed("rave_out"), 0)
	cur_connect.push(max_decoder)
	
	



}



	
	
	
	
	//### TIMBRE CONTROL####
	var timbre_recv = p.getnamed("timbre_receive")
	var timbre_unp = p.newdefault(100, 190, "mc.unpack~",8);
	timbre_unp.rect = [100, 190, 190, 190]

	p.connect(timbre_recv, 0,timbre_unp,  0)

	for (var j = 0; j < n_latents_timbre; j++)
		{
			p.connect(timbre_unp, j,max_decoder,  1 + j)
		}


	cur_connect.push(timbre_unp)
	

	
	
	//### AE LATENTS #### 
	for (var z = 0; z < n_ae_latents; z+=1)
	{
		//

		
		
		p.connect(max_decoder, z, max_codec_decoder, z);
		cur_z.push(cur_connect)
		}
		
	//cur_connect.push(max_codec_decoder);
	

	


	//### TIMBRE CONNECTION ###


	var sigm2l1 = p.getnamed("sigm2l1");
	var sigm2l2 = p.getnamed("sigm2l2");
	var packm2l = p.getnamed("packm2l");


	var unpackl2m = p.getnamed("unpackl2m");
	var snapl2m1= p.getnamed("snapl2m1");
	var snapl2m2 = p.getnamed("snapl2m2");

	var max_m2l =  p.newdefault(2374, 1617, "nn~", model_name, "map2latent");
	var max_l2m =  p.newdefault(2760, 2100	, "nn~", model_name, "latent2map");


	p.connect(sigm2l1, 0 ,max_m2l,  0);
	p.connect(sigm2l2, 0 ,max_m2l,  1);

	for (var j = 0; j < 8; j++)
	{
		p.connect(max_m2l, j ,packm2l,  j)
	}
	
	for (var j = 0; j < 8; j++)
		{
			p.connect(unpackl2m, j ,max_l2m,  j)
		}
		
		p.connect(max_l2m, 0, snapl2m1, 0)
		p.connect(max_l2m, 1, snapl2m2, 0)

	cur_connect.push(max_m2l);
	cur_connect.push(max_l2m);

	outlet(1, "create")
	outlet(0, "bang")
}

function delete()
{
	if (cur_z != 0)
		_delete_current()
}

// delete current model
function _delete_current()
{	
	
	clear_warning_message();
	set_knobs_enabled(1);
	
	
	for (var item in cur_connect){
		p.remove(cur_connect[item])
	}

	// p.remove(cur_decoder)
	// p.remove(timbre_recv)
	// p.remove(timbre_unp)

	var main_plot = p.getnamed("main_plot");
	main_plot.setattr("bkgndpict", "background_transparent.png")

	cur_connect = []
	// for (var t in cur_z)
	// {
	// 	var cur_c = cur_z[t]
	// 	for (var o in cur_c)
	// 		p.remove(cur_c[o])
	// }
	// cur_encoder = 0
	// cur_decoder = 0
	// cur_z = 0
}
