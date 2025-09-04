#!/usr/bin/env python
"""
DECA Flask Server - API for 3D Face Reconstruction
Usage: python deca_server.py
"""

import os
import sys
import json
import uuid
import shutil
import zipfile
import io
from pathlib import Path
from flask import Flask, request, jsonify, send_file
import tempfile
import traceback
from datetime import datetime

# Add DECA to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import DECA modules
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
import torch

app = Flask(__name__)

# Global DECA instance
deca_model = None

def initialize_deca():
    """Initialize DECA model once at startup"""
    global deca_model
    try:
        print("Initializing DECA model...")
        # Use GPU if available, otherwise CPU
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Configure DECA
        deca_cfg.model.use_tex = False  # Disable texture for faster processing
        deca_cfg.model.extract_tex = True
        deca_cfg.rasterizer_type = 'standard'  # Use standard rasterizer
        
        deca_model = DECA(config=deca_cfg, device=device)
        print("DECA model initialized successfully!")
        return True
    except Exception as e:
        print(f"Failed to initialize DECA: {str(e)}")
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': deca_model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/reconstruct', methods=['POST'])
def reconstruct_face():
    """
    Main endpoint for 3D face reconstruction
    
    Expected form data:
    - image: Image file (jpg, png, etc.)
    - output_folder: Name of output folder (optional, defaults to 'results_<uuid>')
    - save_depth: Whether to save depth images (optional, default: true)
    - save_obj: Whether to save 3D mesh files (optional, default: true)
    - save_vis: Whether to save visualization images (optional, default: true)
    - save_kpt: Whether to save keypoints (optional, default: false)
    - save_images: Whether to save component images (optional, default: false)
    """
    
    if deca_model is None:
        return jsonify({'error': 'DECA model not initialized'}), 500
    
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get parameters
        output_folder = request.form.get('output_folder', f'results_{uuid.uuid4().hex[:8]}')
        save_depth = request.form.get('save_depth', 'true').lower() == 'true'
        save_obj = request.form.get('save_obj', 'true').lower() == 'true'
        save_vis = request.form.get('save_vis', 'true').lower() == 'true'
        save_kpt = request.form.get('save_kpt', 'false').lower() == 'true'
        save_images = request.form.get('save_images', 'false').lower() == 'true'
        save_mat = request.form.get('save_mat', 'false').lower() == 'true'
        
        # Create temporary file for input image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_file:
            image_file.save(temp_file.name)
            temp_image_path = temp_file.name
        
        # Create output directory
        output_path = os.path.abspath(output_folder)
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Run DECA reconstruction
            print(f"Processing image: {image_file.filename}")
            print(f"Output folder: {output_path}")
            
            # Import required modules for processing
            from decalib.datasets import datasets
            from decalib.utils import util
            import torch
            import cv2
            import numpy as np
            from scipy.io import savemat
            
            # Load test image
            testdata = datasets.TestData(temp_image_path, iscrop=True, face_detector='fan')
            
            if len(testdata) == 0:
                return jsonify({'error': 'No faces detected in the image'}), 400
            
            # Process the image
            result_files = []
            
            for i in range(len(testdata)):
                name = testdata[i]['imagename']
                images = testdata[i]['image'].to(deca_model.device)[None,...]
                
                with torch.no_grad():
                    codedict = deca_model.encode(images)
                    opdict, visdict = deca_model.decode(codedict)
                
                # Create individual result folder
                result_folder = os.path.join(output_path, name)
                os.makedirs(result_folder, exist_ok=True)
                
                saved_files = []
                
                # Save depth image
                if save_depth:
                    try:
                        depth_image = deca_model.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
                        depth_path = os.path.join(result_folder, f'{name}_depth.jpg')
                        cv2.imwrite(depth_path, util.tensor2image(depth_image[0]))
                        saved_files.append(f'{name}_depth.jpg')
                    except Exception as e:
                        print(f"Warning: Could not save depth image: {e}")
                
                # Save 3D mesh
                if save_obj:
                    try:
                        obj_path = os.path.join(result_folder, f'{name}.obj')
                        deca_model.save_obj(obj_path, opdict)
                        saved_files.extend([f'{name}.obj', f'{name}_detail.obj'])
                    except Exception as e:
                        print(f"Warning: Could not save OBJ files: {e}")
                
                # Save keypoints
                if save_kpt:
                    try:
                        kpt2d_path = os.path.join(result_folder, f'{name}_kpt2d.txt')
                        kpt3d_path = os.path.join(result_folder, f'{name}_kpt3d.txt')
                        np.savetxt(kpt2d_path, opdict['landmarks2d'][0].cpu().numpy())
                        np.savetxt(kpt3d_path, opdict['landmarks3d'][0].cpu().numpy())
                        saved_files.extend([f'{name}_kpt2d.txt', f'{name}_kpt3d.txt'])
                    except Exception as e:
                        print(f"Warning: Could not save keypoints: {e}")
                
                # Save MAT file
                if save_mat:
                    try:
                        mat_path = os.path.join(result_folder, f'{name}.mat')
                        opdict_save = util.dict_tensor2npy(opdict)
                        savemat(mat_path, opdict_save)
                        saved_files.append(f'{name}.mat')
                    except Exception as e:
                        print(f"Warning: Could not save MAT file: {e}")
                
                # Save visualization
                if save_vis:
                    try:
                        vis_path = os.path.join(output_path, f'{name}_vis.jpg')
                        cv2.imwrite(vis_path, deca_model.visualize(visdict))
                        saved_files.append(f'{name}_vis.jpg')
                    except Exception as e:
                        print(f"Warning: Could not save visualization: {e}")
                
                # Save individual images
                if save_images:
                    try:
                        for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                            if vis_name in visdict:
                                img_path = os.path.join(result_folder, f'{name}_{vis_name}.jpg')
                                cv2.imwrite(img_path, util.tensor2image(visdict[vis_name][0]))
                                saved_files.append(f'{name}_{vis_name}.jpg')
                    except Exception as e:
                        print(f"Warning: Could not save individual images: {e}")
                
                result_files.append({
                    'image_name': name,
                    'result_folder': result_folder,
                    'files': saved_files
                })
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            return jsonify({
                'status': 'success',
                'message': 'Face reconstruction completed successfully',
                'output_folder': output_path,
                'results': result_files,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            # Clean up on error
            os.unlink(temp_image_path)
            raise e
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error during reconstruction: {error_msg}")
        traceback.print_exc()
        return jsonify({
            'error': f'Reconstruction failed: {error_msg}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/reconstruct_download', methods=['POST'])
def reconstruct_face_download():
    """
    Endpoint that returns results as a downloadable zip file
    
    Expected form data:
    - image: Image file (jpg, png, etc.)
    - save_depth: Whether to save depth images (optional, default: true)
    - save_obj: Whether to save 3D mesh files (optional, default: true)
    - save_vis: Whether to save visualization images (optional, default: true)
    - save_kpt: Whether to save keypoints (optional, default: false)
    - save_images: Whether to save component images (optional, default: false)
    - save_mat: Whether to save MAT file (optional, default: false)
    """
    
    if deca_model is None:
        return jsonify({'error': 'DECA model not initialized'}), 500
    
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get parameters
        save_depth = request.form.get('save_depth', 'true').lower() == 'true'
        save_obj = request.form.get('save_obj', 'true').lower() == 'true'
        save_vis = request.form.get('save_vis', 'true').lower() == 'true'
        save_kpt = request.form.get('save_kpt', 'false').lower() == 'true'
        save_images = request.form.get('save_images', 'false').lower() == 'true'
        save_mat = request.form.get('save_mat', 'false').lower() == 'true'
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded image to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image_file.filename)[1]) as temp_file:
                image_file.save(temp_file.name)
                temp_image_path = temp_file.name
            
            try:
                # Import required modules
                from decalib.datasets import datasets
                from decalib.utils import util
                import torch
                import cv2
                import numpy as np
                from scipy.io import savemat
                
                # Load test image
                testdata = datasets.TestData(temp_image_path, iscrop=True, face_detector='fan')
                
                if len(testdata) == 0:
                    os.unlink(temp_image_path)
                    return jsonify({'error': 'No faces detected in the image'}), 400
                
                # Create in-memory zip file
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for i in range(len(testdata)):
                        name = testdata[i]['imagename']
                        images = testdata[i]['image'].to(deca_model.device)[None,...]
                        
                        with torch.no_grad():
                            codedict = deca_model.encode(images)
                            opdict, visdict = deca_model.decode(codedict)
                        
                        # Save files to temp directory first, then add to zip
                        result_folder = os.path.join(temp_dir, name)
                        os.makedirs(result_folder, exist_ok=True)
                        
                        # Save depth image
                        if save_depth:
                            try:
                                depth_image = deca_model.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
                                depth_path = os.path.join(result_folder, f'{name}_depth.jpg')
                                cv2.imwrite(depth_path, util.tensor2image(depth_image[0]))
                                zipf.write(depth_path, f'{name}/{name}_depth.jpg')
                            except Exception as e:
                                print(f"Warning: Could not save depth image: {e}")
                        
                        # Save 3D mesh
                        if save_obj:
                            try:
                                obj_path = os.path.join(result_folder, f'{name}.obj')
                                deca_model.save_obj(obj_path, opdict)
                                # Add OBJ files to zip
                                for obj_file in os.listdir(result_folder):
                                    if obj_file.endswith('.obj'):
                                        zipf.write(os.path.join(result_folder, obj_file), f'{name}/{obj_file}')
                            except Exception as e:
                                print(f"Warning: Could not save OBJ files: {e}")
                        
                        # Save keypoints
                        if save_kpt:
                            try:
                                kpt2d_path = os.path.join(result_folder, f'{name}_kpt2d.txt')
                                kpt3d_path = os.path.join(result_folder, f'{name}_kpt3d.txt')
                                np.savetxt(kpt2d_path, opdict['landmarks2d'][0].cpu().numpy())
                                np.savetxt(kpt3d_path, opdict['landmarks3d'][0].cpu().numpy())
                                zipf.write(kpt2d_path, f'{name}/{name}_kpt2d.txt')
                                zipf.write(kpt3d_path, f'{name}/{name}_kpt3d.txt')
                            except Exception as e:
                                print(f"Warning: Could not save keypoints: {e}")
                        
                        # Save MAT file
                        if save_mat:
                            try:
                                mat_path = os.path.join(result_folder, f'{name}.mat')
                                opdict_save = util.dict_tensor2npy(opdict)
                                savemat(mat_path, opdict_save)
                                zipf.write(mat_path, f'{name}/{name}.mat')
                            except Exception as e:
                                print(f"Warning: Could not save MAT file: {e}")
                        
                        # Save visualization
                        if save_vis:
                            try:
                                vis_path = os.path.join(result_folder, f'{name}_vis.jpg')
                                cv2.imwrite(vis_path, deca_model.visualize(visdict))
                                zipf.write(vis_path, f'{name}_vis.jpg')
                            except Exception as e:
                                print(f"Warning: Could not save visualization: {e}")
                        
                        # Save individual images
                        if save_images:
                            try:
                                for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images', 'landmarks2d']:
                                    if vis_name in visdict:
                                        img_path = os.path.join(result_folder, f'{name}_{vis_name}.jpg')
                                        cv2.imwrite(img_path, util.tensor2image(visdict[vis_name][0]))
                                        zipf.write(img_path, f'{name}/{name}_{vis_name}.jpg')
                            except Exception as e:
                                print(f"Warning: Could not save individual images: {e}")
                
                # Clean up temporary file
                os.unlink(temp_image_path)
                
                # Prepare zip for sending
                zip_buffer.seek(0)
                
                # Generate filename based on original image name
                base_name = os.path.splitext(image_file.filename)[0]
                zip_filename = f'{base_name}_reconstruction.zip'
                
                return send_file(
                    zip_buffer,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name=zip_filename
                )
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)
                raise e
                
    except Exception as e:
        error_msg = str(e)
        print(f"Error during reconstruction: {error_msg}")
        traceback.print_exc()
        return jsonify({
            'error': f'Reconstruction failed: {error_msg}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return """
    <h1>DECA 3D Face Reconstruction API</h1>
    
    <h2>Endpoints:</h2>
    
    <h3>POST /reconstruct</h3>
    <p>Upload an image for 3D face reconstruction (saves files on server)</p>
    
    <h4>Form Parameters:</h4>
    <ul>
        <li><strong>image</strong> (required): Image file (jpg, png, etc.)</li>
        <li><strong>output_folder</strong> (optional): Output folder name</li>
        <li><strong>save_depth</strong> (optional): Save depth images (true/false, default: true)</li>
        <li><strong>save_obj</strong> (optional): Save 3D mesh files (true/false, default: true)</li>
        <li><strong>save_vis</strong> (optional): Save visualization (true/false, default: true)</li>
        <li><strong>save_kpt</strong> (optional): Save keypoints (true/false, default: false)</li>
        <li><strong>save_images</strong> (optional): Save component images (true/false, default: false)</li>
        <li><strong>save_mat</strong> (optional): Save MAT file (true/false, default: false)</li>
    </ul>
    
    <h3>POST /reconstruct_download</h3>
    <p>Upload an image for 3D face reconstruction (returns zip file for download)</p>
    
    <h4>Form Parameters:</h4>
    <ul>
        <li><strong>image</strong> (required): Image file (jpg, png, etc.)</li>
        <li><strong>save_depth</strong> (optional): Save depth images (true/false, default: true)</li>
        <li><strong>save_obj</strong> (optional): Save 3D mesh files (true/false, default: true)</li>
        <li><strong>save_vis</strong> (optional): Save visualization (true/false, default: true)</li>
        <li><strong>save_kpt</strong> (optional): Save keypoints (true/false, default: false)</li>
        <li><strong>save_images</strong> (optional): Save component images (true/false, default: false)</li>
        <li><strong>save_mat</strong> (optional): Save MAT file (true/false, default: false)</li>
    </ul>
    
    <h4>Example curl commands:</h4>
    <pre>
# Save on server:
curl -X POST http://localhost:5000/reconstruct \\
  -F "image=@path/to/image.jpg" \\
  -F "output_folder=my_results" \\
  -F "save_depth=true" \\
  -F "save_obj=true"

# Download as zip:
curl -X POST http://localhost:5000/reconstruct_download \\
  -F "image=@path/to/image.jpg" \\
  -F "save_depth=true" \\
  -F "save_obj=true" \\
  -F "save_kpt=true" \\
  -F "save_images=true" \\
  -F "save_mat=true" \\
  -o reconstruction_results.zip
    </pre>
    
    <h3>GET /health</h3>
    <p>Check server health and model status</p>
    """

if __name__ == '__main__':
    print("Starting DECA Server...")
    
    # Initialize DECA model
    if not initialize_deca():
        print("Failed to initialize DECA model. Exiting.")
        sys.exit(1)
    
    # Start Flask server
    print("Server starting on http://localhost:5000")
    print("Access the API documentation at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)