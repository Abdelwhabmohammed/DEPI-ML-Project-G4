import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_img_array(img_path, size=(299, 299)):
    """Load and preprocess image for InceptionV3."""
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    return array


def make_layercam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate LayerCAM heatmap.
    
    Args:
        img_array: Preprocessed image array
        model: Trained model
        last_conv_layer_name: Name of the last convolutional layer
        pred_index: Index of the predicted class (if None, use argmax)
        
    Returns:
        Heatmap array
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for the input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # LayerCAM computation
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Multiply each channel by its importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_layercam(img_path, heatmap, output_path, alpha=0.4):
    """
    Overlay heatmap on original image and save.
    
    Args:
        img_path: Path to original image
        heatmap: Generated heatmap
        output_path: Path to save output
        alpha: Transparency of heatmap overlay
    """
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.uint8(superimposed_img)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap)
    axes[1].set_title('LayerCAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(superimposed_img)
    axes[2].set_title('Superimposed')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.show()
    
    print(f"LayerCAM visualization saved to {output_path}")


def main():
    """Main function for LayerCAM visualization."""
    print("\n" + "="*60)
    print("LayerCAM Visualization for InceptionV3")
    print("="*60 + "\n")
    
    # Load InceptionV3 model
    print("Loading InceptionV3 model...")
    model = InceptionV3(weights='imagenet')
    print("Model loaded successfully.\n")
    
    # Get image path
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        print("Please provide an image path as argument:")
        print("  python scripts/LayerCAM.py <image_path>")
        return
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
    
    # Preprocess image
    print(f"Processing image: {img_path}")
    img_array = get_img_array(img_path)
    
    # Make prediction
    preds = model.predict(img_array)
    print("\nTop 3 Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decode_predictions(preds, top=3)[0]):
        print(f"  {i+1}. {label}: {score*100:.2f}%")
    
    # Generate LayerCAM heatmap
    last_conv_layer_name = 'mixed10'
    print(f"\nGenerating LayerCAM using layer: {last_conv_layer_name}")
    
    heatmap = make_layercam_heatmap(img_array, model, last_conv_layer_name)
    
    # Save visualization
    output_dir = './visualizations'
    os.makedirs(output_dir, exist_ok=True)
    
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    output_path = os.path.join(output_dir, f'{img_basename}_layercam.png')
    
    save_and_display_layercam(img_path, heatmap, output_path)
    
    print(f"\n{'='*60}")
    print("LayerCAM visualization completed!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()