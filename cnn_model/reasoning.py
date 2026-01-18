import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from glob import glob
# ## Model and Device Setup

model_path = "model_final_weights.pth"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# ## Disease Classes

diseases = ['cataract','diabetic_retinopathy','glaucoma', 'normal']
class ResBlock(nn.Module):
    '''A resnet block with skip connection'''
    def __init__(self, in_channels:int, out_channels:int, stride:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if (in_channels!=out_channels or stride!=1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x:torch.tensor)->torch.tensor:
        out = torch.relu(self.batchnorm1(self.conv1(x)))
        out = self.batchnorm2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18(nn.Module):
    '''A ResNet18 model'''
    def __init__(self, num_classes:int=8):  # Set to your number of classes
        super().__init__()
        self.in_channels=64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)

        self.layer1 = self.make_blocks(ResBlock, 64, 2, 1)
        self.layer2 = self.make_blocks(ResBlock, 128, 2, 2)
        self.layer3 = self.make_blocks(ResBlock, 256, 2, 2)
        self.layer4 = self.make_blocks(ResBlock, 512, 2, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
    def make_blocks(self, block:ResBlock, out_channels:int, num_blocks:int, stride:int):
        '''make a residual block'''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in strides:
            layers.append(block(self.in_channels, out_channels, stride=i))
            self.in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self, x:torch.tensor)->torch.tensor:
        out = self.batchnorm1(self.conv1(x))
        out = F.max_pool2d(torch.relu(out), 2)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.layer1(out)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.layer2(out)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.layer3(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.layer4(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.fc(out)
        return out
# ## Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), output, target_class

# ## Reasoning Generator
class ReasoningGenerator:
    def __init__(self, disease_names):
        self.disease_names = disease_names

        # Clinical information for each disease
        self.disease_info = {
            'Glaucoma': {
                'features': ['increased cup-to-disc ratio', 'optic disc pallor', 'RNFL thinning'],
                'locations': ['optic disc', 'peripapillary region']
            },
            'Normal': {
                'features': ['healthy optic disc', 'clear macula', 'normal vessel pattern'],
                'locations': ['throughout the retina']
            },
            'Diabetic Retinopathy': {
                'features': ['microaneurysms', 'hemorrhages', 'hard exudates', 'cotton wool spots'],
                'locations': ['posterior pole', 'macula', 'peripheral retina']
            },
            'Cataract': {
                'features': ['lens opacity', 'reduced fundus clarity', 'decreased red reflex'],
                'locations': ['lens', 'overall image clarity']
            }

        }
    def analyze_cam(self, cam):
        """Analyze spatial attention pattern from Grad-CAM"""
        h, w = cam.shape
        center_h, center_w = h // 3, w // 3

        # Region analysis
        center = cam[center_h:2*center_h, center_w:2*center_w].mean()
        superior = cam[:center_h, :].mean()
        inferior = cam[2*center_h:, :].mean()
        nasal = cam[:, :center_w].mean()
        temporal = cam[:, 2*center_w:].mean()
        periphery = np.concatenate([
            cam[:center_h, :].flatten(),
            cam[2*center_h:, :].flatten(),
            cam[:, :center_w].flatten(),
            cam[:, 2*center_w:].flatten()
        ]).mean()

        # Attention statistics
        high_attention_ratio = (cam > 0.7).sum() / cam.size
        attention_spread = cam.std()
        max_attention = cam.max()

        return {
            'center': center,
            'superior': superior,
            'inferior': inferior,
            'nasal': nasal,
            'temporal': temporal,
            'periphery': periphery,
            'high_attention_ratio': high_attention_ratio,
            'attention_spread': attention_spread,
            'max_attention': max_attention
        }

    def generate(self, disease_name, confidence, cam_analysis, all_probs):
        """Generate comprehensive medical reasoning"""
        parts = []

        # 1. Primary Diagnosis
        if confidence > 0.85:
            conf_desc = "high confidence"
        elif confidence > 0.65:
            conf_desc = "moderate confidence"
        else:
            conf_desc = "low confidence"

        parts.append(f"**PRIMARY DIAGNOSIS:** {disease_name}")
        parts.append(f"**Confidence Level:** {conf_desc} ({confidence*100:.1f}%)")

        # 2. Differential Diagnosis
        if confidence < 0.90:
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            top_alternatives = [f"{name} ({prob*100:.1f}%)" 
                              for name, prob in sorted_probs[1:4] if prob > 0.05]
            if top_alternatives:
                parts.append(f"**Differential Diagnoses:** {', '.join(top_alternatives)}")

        # 3. Anatomical Regions of Interest
        regions_found = []
        if cam_analysis['center'] > 0.5:
            regions_found.append("macular region")
        if cam_analysis['superior'] > 0.4 or cam_analysis['inferior'] > 0.4:
            regions_found.append("arcuate nerve fiber bundles")
        if max(cam_analysis['nasal'], cam_analysis['temporal']) > 0.4:
            if cam_analysis['nasal'] > cam_analysis['temporal']:
                regions_found.append("nasal retina (optic disc)")
            else:
                regions_found.append("temporal retina")
        if cam_analysis['periphery'] > 0.35:
            regions_found.append("peripheral retina")

        if regions_found:
            parts.append(f"**Key Anatomical Regions:** {', '.join(regions_found)}")

        # 4. Clinical Features (based on disease)
        if disease_name in self.disease_info:
            info = self.disease_info[disease_name]
            parts.append(f"**Expected Clinical Features:** {', '.join(info['features'][:3])}")

        # 5. Attention Pattern Analysis
        if cam_analysis['high_attention_ratio'] > 0.15:
            pattern_desc = "Multiple focal areas of high attention suggest presence of several pathological features"
        elif cam_analysis['high_attention_ratio'] > 0.05:
            pattern_desc = "Localized attention pattern indicates concentrated pathological changes"
        else:
            pattern_desc = "Diffuse attention pattern across the fundus"

        parts.append(f"**Attention Pattern:** {pattern_desc}")

        # 6. Attention Spread Interpretation
        if cam_analysis['attention_spread'] > 0.25:
            parts.append(f"**Spatial Distribution:** High attention variability (σ={cam_analysis['attention_spread']:.3f}) "
                        "suggests heterogeneous pathological changes")
        else:
            parts.append(f"**Spatial Distribution:** Focused attention (σ={cam_analysis['attention_spread']:.3f}) "
                        "indicates localized disease features")

        # 7. Clinical Correlation
        if disease_name != 'Normal':
            if cam_analysis['center'] > cam_analysis['periphery'] * 1.5:
                parts.append("**Clinical Correlation:** Central retinal involvement is consistent with the diagnosis")
            elif cam_analysis['periphery'] > cam_analysis['center'] * 1.2:
                parts.append("**Clinical Correlation:** Peripheral retinal changes are the predominant finding")

        # 8. Recommendation
        if confidence > 0.85 and disease_name == 'Normal':
            recommendation = "No significant pathological findings detected. Continue routine screening as per guidelines."
        elif confidence > 0.75:
            recommendation = f"Findings suggest {disease_name}. Recommend comprehensive ophthalmic examination with OCT and clinical correlation."
        elif confidence > 0.50:
            recommendation = f"Possible {disease_name} detected. Further diagnostic workup including multimodal imaging recommended."
        else:
            recommendation = "Uncertain diagnosis. Recommend repeat imaging with better quality and comprehensive eye examination."

        parts.append(f"**Clinical Recommendation:** {recommendation}")

        # 9. Disclaimer
        parts.append("\n⚠️ **IMPORTANT DISCLAIMER:** This is an AI-assisted analysis tool for educational and screening purposes only. "
                    "It does NOT replace professional medical diagnosis. All findings must be verified by a qualified ophthalmologist "
                    "through comprehensive clinical examination.")

        return "\n\n".join(parts)

# ## Prediction Function
def predict_with_reasoning(model_path, image_path, disease_names, device='cuda'):
    """
    Complete prediction with reasoning for your trained model
    """
    # Load model
    num_classes = len(diseases)
    model = ResNet18(num_classes=num_classes)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Preprocessing 
    preprocess = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.4203, 0.2800, 0.1714),
                    std=(0.2932, 0.2165, 0.1632))
    ])

    # Load image
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(original_image).unsqueeze(0).to(device)

    # Grad-CAM on layer4 (last residual block)
    target_layer = model.layer4[-1]  # Last block in layer4
    grad_cam = GradCAM(model, target_layer)
    cam, output, pred_class = grad_cam.generate_cam(input_tensor)

    # Get probabilities
    probs = F.softmax(output, dim=1)[0].detach().cpu().numpy()
    all_probs = {disease_names[i]: float(probs[i]) for i in range(len(disease_names))}

    # Generate reasoning
    reasoning_gen = ReasoningGenerator(disease_names)
    cam_analysis = reasoning_gen.analyze_cam(cam)
    reasoning = reasoning_gen.generate(
        disease_names[pred_class],
        float(probs[pred_class]),
        cam_analysis,
        all_probs
    )

    # Visualize
    visualize(original_image, cam, disease_names[pred_class], 
             probs[pred_class], reasoning, all_probs, image_path)

    return {
        'disease': disease_names[pred_class],
        'confidence': float(probs[pred_class]),
        'probabilities': all_probs,
        'reasoning': reasoning,
        'cam': cam
    }

# ## Visualization

def visualize(original_img, cam, disease, confidence, reasoning, all_probs, img_path):
    """Create comprehensive visualization"""
    original_np = np.array(original_img)
    h, w = original_np.shape[:2]

    # Create heatmap
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (heatmap * 0.4 + original_np * 0.6).astype(np.uint8)

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.5, 1.5, 1], hspace=0.3, wspace=0.3)

    fig.suptitle(f'Eye Disease Detection with Clinical Reasoning\nImage: {os.path.basename(img_path)}', 
                 fontsize=16, fontweight='bold', y=0.98)

    # Row 1: Images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_np)
    ax1.set_title('Original Fundus Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(heatmap)
    ax2.set_title('Grad-CAM Attention Map', fontsize=12, fontweight='bold')
    ax2.axis('off')
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax2, fraction=0.046)
    cbar.set_label('Attention Intensity', fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.set_title(f'Overlay Visualization\nDiagnosis: {disease}\nConfidence: {confidence*100:.1f}%', 
                 fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Row 2: Probability bars
    ax4 = fig.add_subplot(gs[1, :2])
    sorted_diseases = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    y_pos = np.arange(len(sorted_diseases))
    probs_values = [p for _, p in sorted_diseases]
    disease_labels = [d for d, _ in sorted_diseases]

    colors = ['darkred' if i == 0 else 'steelblue' for i in range(len(sorted_diseases))]
    bars = ax4.barh(y_pos, probs_values, color=colors, alpha=0.8)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(disease_labels, fontsize=10)
    ax4.set_xlabel('Probability', fontsize=11, fontweight='bold')
    ax4.set_title('Disease Classification Probabilities', fontsize=12, fontweight='bold')
    ax4.set_xlim([0, 1])
    ax4.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (bar, prob) in enumerate(zip(bars, probs_values)):
        ax4.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', fontsize=9)

    # Attention distribution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(cam_resized, cmap='hot')
    ax5.set_title('Attention Intensity Map', fontsize=12, fontweight='bold')
    ax5.axis('off')

    # Row 3: Clinical reasoning (full width)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    ax6.text(0.02, 0.95, 'CLINICAL REASONING AND ANALYSIS:', 
            fontsize=13, fontweight='bold', va='top', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
    ax6.text(0.02, 0.80, reasoning, fontsize=9, va='top', wrap=True,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.2))

    plt.subplots_adjust()

    # Save figure
    output_filename = f'prediction_{os.path.splitext(os.path.basename(img_path))[0]}.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved as: {output_filename}")
    plt.show()

    # Print reasoning to console
    print("\n" + "="*80)
    print("CLINICAL REASONING REPORT")
    print("="*80)
    print(f"\nImage: {img_path}")
    print(f"\n{reasoning}")
    print("\n" + "="*80)

def batch_predict(model_path, image_folder, disease_names, device='cuda', max_images=None):
    """Process multiple images"""
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_folder, ext)))
        image_files.extend(glob(os.path.join(image_folder, ext.upper())))

    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images to process\n")

    results_summary = []

    for idx, img_path in enumerate(image_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing Image {idx}/{len(image_files)}: {os.path.basename(img_path)}")
        print(f"{'='*80}")

        try:
            result = predict_with_reasoning(model_path, img_path, disease_names, device)
            results_summary.append({
                'image': os.path.basename(img_path),
                'disease': result['disease'],
                'confidence': result['confidence']
            })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results_summary.append({
                'image': os.path.basename(img_path),
                'disease': 'ERROR',
                'confidence': 0.0
            })

    # Print summary
    print("\n\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    for r in results_summary:
        print(f"{r['image']:40s} -> {r['disease']:25s} ({r['confidence']*100:.1f}%)")
    print("="*80)

    return results_summary

#Test Image
Test_Image_Path = '/kaggle/input/eye-diseases-classification/dataset'


# ## Main Execution


if __name__ == "__main__":
    print("="*80)
    print("EYE DISEASE DETECTION WITH REASONING")
    print("="*80)

    # Single image prediction
    result = predict_with_reasoning(
        model_path=model_path,
        image_path=Test_Image_Path + '/cataract/1084_right.jpg',  
        disease_names=diseases,
        device=DEVICE
    )

    # Uncomment for batch processing:
    # results = batch_predict(
    #     model_path=MODEL_PATH,
    #     image_folder=TEST_IMAGE_PATH,
    #     disease_names=DISEASE_NAMES,
    #     device=DEVICE,
    #     max_images=5  # Process first 5 images
    # )

