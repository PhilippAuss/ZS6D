import os,sys
# sys.path.append(".")
# sys.path.append("./bop_toolkit")


import math
from plyfile import PlyData, PlyElement
import numpy as np
import itertools
# from tools import bop_io
# from bop_toolkit.bop_toolkit_lib import inout,dataset_params
from numpy.lib import recfunctions

def get_xyz_max(fn_read):
    plydata = PlyData.read(fn_read)
    #x,y,z : embbedding to RGB
    x_ct = np.mean(plydata.elements[0].data['x'])    
    x_abs = np.max(np.abs(plydata.elements[0].data['x']-x_ct))    
    y_ct = np.mean(plydata.elements[0].data['y'])    
    y_abs = np.max(np.abs(plydata.elements[0].data['y']-y_ct))   
    
    z_ct = np.mean(plydata.elements[0].data['z'])    
    z_abs = np.max(np.abs(plydata.elements[0].data['z']-z_ct))   
    
    return x_abs,y_abs,z_abs,x_ct,y_ct,z_ct


def _add_field(field: str, data: np.ndarray, field_type='u1'):
    # Create a new dtype that includes the existing fields plus the 'red' field
    new_dtype = np.dtype(data.dtype.descr + [(field, field_type)])
    # Initialize a new array with the same shape as the original data, but with the new dtype
    new_data = np.zeros(data.shape, dtype=new_dtype)
    # Copy the data from the original fields to the new array
    for name in data.dtype.names:
        new_data[name] = data[name]
        
    # Initialize the 'red' field with zeros
    # .shape[0] ensures we're creating a 1D array of zeros with the correct length
    new_data[field] = np.zeros(data.shape[0])
    
    return new_data

def _add_fields_to_PlyData(ply_data, new_fields=[('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]):
    # Check the existing dtype of your vertex data
    existing_dtype = ply_data['vertex'].data.dtype.descr
    
    # Filter out any new fields that already exist to avoid duplication
    existing_field_names = [field[0] for field in existing_dtype]
    new_fields_filtered = [field for field in new_fields if field[0] not in existing_field_names]
    
    # Create a new dtype that includes both the existing and new fields
    new_dtype = existing_dtype + new_fields_filtered
    
    # Create a new numpy array for the vertex data with the new dtype
    new_vertex_data = np.zeros(ply_data['vertex'].count, dtype=new_dtype)
    
    # Copy existing vertex data into the new array
    for name in ply_data['vertex'].data.dtype.names:
        new_vertex_data[name] = ply_data['vertex'].data[name]
    
    # Initialize new fields with default values
    # for field_name, _ in new_fields_filtered:
    #     new_vertex_data[field_name] = 0 
    
    # Create a new PlyElement for the modified vertex data
    new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    
    # Reconstruct the PlyData object with the new vertex element
    # This involves keeping all other elements unchanged
    new_elements = [new_vertex_element if el.name == 'vertex' else el for el in ply_data.elements]
    modified_ply_data = PlyData(new_elements, text=ply_data.text)
    
    return modified_ply_data
    

def convert_unique(fn_read,fn_write,center_x=True,center_y=True,center_z=True):
    plydata = PlyData.read(fn_read)

    #x,y,z : embbedding to RGB
    x_ct = np.mean(plydata.elements[0].data['x'])    
    if not(center_x):
        x_ct=0
    x_abs = np.max(np.abs(plydata.elements[0].data['x']-x_ct))
    
    y_ct = np.mean(plydata.elements[0].data['y'])    
    if not(center_y):
        y_ct=0
    y_abs = np.max(np.abs(plydata.elements[0].data['y']-y_ct))    
    
    z_ct = np.mean(plydata.elements[0].data['z'])    
    if not(center_z):
        z_ct=0
    z_abs = np.max(np.abs(plydata.elements[0].data['z']-z_ct))    
    n_vert = plydata.elements[0].data['x'].shape[0]
    
    if 'red' not in plydata.elements[0].data.dtype.names:
        plydata = _add_fields_to_PlyData(plydata)
           
    for i in range(n_vert):
        r=(plydata.elements[0].data['x'][i]-x_ct)/x_abs #-1 to 1
        r = (r+1)/2 #0 to 2 -> 0 to 1        
        g=(plydata.elements[0].data['y'][i]-y_ct)/y_abs
        g = (g+1)/2
        b=(plydata.elements[0].data['z'][i]-z_ct)/z_abs
        b = (b+1)/2
        plydata.elements[0].data['red'][i]=r*255
        plydata.elements[0].data['green'][i]=g*255
        plydata.elements[0].data['blue'][i]=b*255
    plydata.write(fn_write)        
    return x_abs,y_abs,z_abs,x_ct,y_ct,z_ct

def rmfield( a, *fieldnames_to_remove ):
    return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]



# if(len(sys.argv)<2):
#     print("python3 tools/2_1_ply_file_to_3d_coord_model.py [cfg_fn] [dataset_name]")

# cfg_fn =sys.argv[1]
# cfg = inout.load_json(cfg_fn)

# dataset = sys.argv[2]
# bop_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,gts,cam_param_global = bop_io.get_dataset(cfg,dataset)


# if not(os.path.exists(bop_dir + "/models_xyz/")):
#     os.makedirs(bop_dir + "/models_xyz/")
# norm_factor = bop_dir+"/models_xyz/"+"norm_factor.json"
# param={}


# for m_id,model_ply in enumerate(model_plys):
#     model_id = model_ids[m_id]
#     m_info = model_info['{}'.format(model_id)]
#     keys = m_info.keys()
#     sym_continous = [0,0,0,0,0,0]
#     center_x = center_y = center_z = True    
#     #if('symmetries_discrete' in keys): #use this when objects are centered already
#     #    center_x = center_y = center_z = False
#     #    print("keep origins of the object when it has symmetric poses")    
#     fn_read = model_ply
#     fname = model_ply.split("/")[-1]
#     obj_id = int(fname[4:-4])
#     fn_write = bop_dir + "/models_xyz/" + fname    
#     x_abs,y_abs,z_abs,x_ct,y_ct,z_ct = convert_unique(fn_read,fn_write,center_x=center_x,center_y=center_y,center_z=center_z)
#     print(obj_id,x_abs,y_abs,z_abs,x_ct,y_ct,z_ct)
#     param[int(obj_id)]={'x_scale':float(x_abs),'y_scale':float(y_abs),'z_scale':float(z_abs),'x_ct':float(x_ct),'y_ct':float(y_ct),'z_ct':float(z_ct)}

# inout.save_json(norm_factor,param)