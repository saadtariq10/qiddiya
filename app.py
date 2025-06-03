# import streamlit as st
# from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
# from datetime import datetime
# import cv2
# import numpy as np
# import io

# # CONFIG
# FONT_PATH = "fonts/custom_font.ttf"
# FONT_SIZE = 24
# FONT_COLOR = (255, 251, 223)  # Hex #FFFBDF
# QR_SIZE = (158, 155)

# st.title("Qiddiya Work Access Pass Generator")

# # Create two columns for input fields
# col1, col2 = st.columns(2)

# with col1:
#     name = st.text_input("Full Name", key="name").upper()
#     company = st.text_input("Company Name", key="company").upper()
#     id_number = st.text_input("ID Number", key="id_number")

# with col2:
#     issue_date = st.date_input(
#         "Issue Date",
#         key="issue_date"
#     )
#     expiry_date = st.date_input(
#         "Expiry Date",
#         min_value=issue_date,
#         key="expiry_date"
#     )

# uploaded_photo = st.file_uploader("Upload Existing Pass (to extract QR code)", type=["jpg", "jpeg", "png"])

# def draw_centered(draw, text, y, font, image_width):
#     bbox = font.getbbox(text)
#     text_width = bbox[2] - bbox[0]
#     x = (image_width - text_width) // 2
#     draw.text((x, y), text, font=font, fill=FONT_COLOR)

# if uploaded_photo:
#     photo = Image.open(uploaded_photo).convert("RGB")
    
#     # Convert PIL image to OpenCV format
#     open_cv_image = np.array(photo)
#     open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR

#     # QR Code Detection using OpenCV
#     detector = cv2.QRCodeDetector()
#     data, points, _ = detector.detectAndDecode(open_cv_image)
#     qr_image = None
#     if points is not None:
#         # Get bounding box and crop QR code
#         points = points[0]
#         x1 = int(min(points[:, 0]))
#         y1 = int(min(points[:, 1]))
#         x2 = int(max(points[:, 0]))
#         y2 = int(max(points[:, 1]))
#         qr_image = photo.crop((x1, y1, x2, y2)).resize(QR_SIZE)
        
#         if all([name, company, id_number]) and st.button("Generate Pass"):
#             template = Image.open("template.png").convert("RGB")
#             draw = ImageDraw.Draw(template)
#             font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
            
#             # Format dates
#             formatted_issue_date = issue_date.strftime("%d/%m/%Y")
#             formatted_expiry_date = expiry_date.strftime("%d/%m/%Y")
            
#             # Issue date
#             draw_centered(draw, formatted_issue_date, 240, font, template.width)
            
#             # Center QR code - moved slightly upward
#             if qr_image:
#                 qr_pos_x = (template.width - QR_SIZE[0]) // 2
#                 qr_pos_y = 471  # Moved up from 550 to 500
#                 template.paste(qr_image, (qr_pos_x, qr_pos_y))
            
#             # Text below QR code - moved down by increasing initial y_pos
#             y_pos = qr_pos_y + QR_SIZE[1] + 60  # Increased from 30 to 50 to move everything down
#             draw_centered(draw, f"EX {formatted_expiry_date}", y_pos, font, template.width)
#             y_pos += 30
#             draw_centered(draw, name, y_pos, font, template.width)
#             y_pos += 30
#             draw_centered(draw, company, y_pos, font, template.width)
#             y_pos += 30
#             draw_centered(draw, id_number, y_pos, font, template.width)
            
#             # Show preview
#             st.image(template, caption="Preview of Access Pass", use_container_width=True)
            
#             # Download button
#             buf = io.BytesIO()
#             template.save(buf, format="PNG")
#             st.download_button(
#                 "Download Access Pass",
#                 data=buf.getvalue(),
#                 file_name=f"qiddiya_pass_{name.lower().replace(' ', '_')}.png",
#                 mime="image/png"
#             )
#     else:
#         st.error("No QR code detected in the uploaded image. Please upload a valid pass with QR code.")



#!/usr/bin/env python3
"""
Robust Qiddiya Work Access Pass Generator
Comprehensive error handling with multiple fallback options
Removed cv2 dependency - uses PIL/Pillow with QR detection alternatives
"""

import streamlit as st
import os
import sys
import logging
import traceback
import io
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Any
import tempfile

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
@st.cache_resource
def setup_logging():
    """Setup comprehensive logging"""
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_filename = f"qiddiya_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_path = log_dir / log_filename
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    except Exception as e:
        # Fallback to basic logging if directory creation fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not setup file logging: {e}")
        return logger

logger = setup_logging()

class SafeImageProcessor:
    """Safe image processing with multiple fallback methods"""
    
    def __init__(self):
        self.pil_available = False
        self.qr_libs = []
        self.setup_libraries()
    
    def setup_libraries(self):
        """Initialize available libraries with error handling"""
        # Check PIL availability
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
            self.pil_available = True
            self.Image = Image
            self.ImageDraw = ImageDraw
            self.ImageFont = ImageFont
            logger.info("PIL/Pillow loaded successfully")
        except ImportError as e:
            logger.error(f"PIL/Pillow not available: {e}")
            st.error("PIL/Pillow is required but not installed. Please install: pip install Pillow")
            return False
        
        # Check QR code detection libraries (multiple options)
        qr_options = [
            ('pyzbar', self._try_pyzbar),
            ('qrcode', self._try_qrcode),
            ('opencv-python', self._try_opencv),
            ('pyzbar-fallback', self._try_manual_qr)
        ]
        
        for lib_name, init_func in qr_options:
            try:
                if init_func():
                    self.qr_libs.append(lib_name)
                    logger.info(f"QR library {lib_name} loaded successfully")
            except Exception as e:
                logger.warning(f"QR library {lib_name} failed to load: {e}")
        
        if not self.qr_libs:
            logger.warning("No QR detection libraries available - using manual detection")
            self.qr_libs.append('manual')
        
        return True
    
    def _try_pyzbar(self):
        """Try to initialize pyzbar"""
        try:
            from pyzbar import pyzbar
            from pyzbar.pyzbar import decode
            self.pyzbar = pyzbar
            self.decode = decode
            return True
        except ImportError:
            return False
    
    def _try_qrcode(self):
        """Try to initialize qrcode library"""
        try:
            import qrcode
            from qrcode.image.pil import PilImage
            self.qrcode = qrcode
            return True
        except ImportError:
            return False
    
    def _try_opencv(self):
        """Try to initialize OpenCV"""
        try:
            import cv2
            import numpy as np
            self.cv2 = cv2
            self.np = np
            return True
        except ImportError:
            return False
    
    def _try_manual_qr(self):
        """Always available manual QR detection"""
        return True
    
    def safe_image_open(self, image_source) -> Optional[Any]:
        """Safely open image from various sources"""
        try:
            if hasattr(image_source, 'read'):
                # File-like object
                image_source.seek(0)
                image = self.Image.open(image_source).convert("RGB")
            elif isinstance(image_source, str):
                # File path
                if not os.path.exists(image_source):
                    logger.error(f"File not found: {image_source}")
                    return None
                image = self.Image.open(image_source).convert("RGB")
            else:
                # PIL Image object
                image = image_source.convert("RGB")
            
            logger.info(f"Image opened successfully: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            return None
    
    def detect_qr_code(self, image) -> Optional[Any]:
        """Detect QR code using available methods"""
        if not image:
            return None
        
        for lib in self.qr_libs:
            try:
                if lib == 'pyzbar' and hasattr(self, 'decode'):
                    return self._detect_qr_pyzbar(image)
                elif lib == 'opencv-python' and hasattr(self, 'cv2'):
                    return self._detect_qr_opencv(image)
                elif lib == 'manual':
                    return self._detect_qr_manual(image)
            except Exception as e:
                logger.warning(f"QR detection failed with {lib}: {e}")
                continue
        
        logger.warning("All QR detection methods failed")
        return None
    
    def _detect_qr_pyzbar(self, image):
        """Detect QR using pyzbar"""
        try:
            import numpy as np
            img_array = np.array(image)
            codes = self.decode(img_array)
            
            if codes:
                code = codes[0]
                # Get bounding box
                rect = code.rect
                x, y, w, h = rect.left, rect.top, rect.width, rect.height
                qr_image = image.crop((x, y, x + w, y + h))
                logger.info("QR code detected with pyzbar")
                return qr_image
        except Exception as e:
            logger.error(f"Pyzbar QR detection failed: {e}")
        return None
    
    def _detect_qr_opencv(self, image):
        """Detect QR using OpenCV"""
        try:
            import numpy as np
            open_cv_image = np.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR
            
            detector = self.cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(open_cv_image)
            
            if points is not None:
                points = points[0]
                x1 = int(min(points[:, 0]))
                y1 = int(min(points[:, 1]))
                x2 = int(max(points[:, 0]))
                y2 = int(max(points[:, 1]))
                qr_image = image.crop((x1, y1, x2, y2))
                logger.info("QR code detected with OpenCV")
                return qr_image
        except Exception as e:
            logger.error(f"OpenCV QR detection failed: {e}")
        return None
    
    def _detect_qr_manual(self, image):
        """Manual QR detection using image analysis"""
        try:
            # Simple approach: look for square regions with high contrast
            import numpy as np
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # This is a simplified approach - in practice, you'd want more sophisticated detection
            # For now, we'll assume the QR code is in a specific region or ask user to crop manually
            logger.info("Using manual QR detection - may require user intervention")
            return image  # Return full image as fallback
        except Exception as e:
            logger.error(f"Manual QR detection failed: {e}")
        return None

# Initialize processor
@st.cache_resource
def get_image_processor():
    """Get cached image processor instance"""
    processor = SafeImageProcessor()
    if not processor.pil_available:
        st.stop()
    return processor

processor = get_image_processor()

# Configuration with safe defaults
class Config:
    """Safe configuration management"""
    
    def __init__(self):
        self.font_paths = [
            "fonts/custom_font.ttf",
            "fonts/arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "C:/Windows/Fonts/arial.ttf",  # Windows
        ]
        self.font_size = 24
        self.font_color = (255, 251, 223)  # #FFFBDF
        self.qr_size = (158, 155)
        self.template_paths = [
            "template.png",
            "templates/template.png",
            "assets/template.png"
        ]
    
    def get_font(self, size=None):
        """Get available font with fallbacks"""
        size = size or self.font_size
        
        for font_path in self.font_paths:
            try:
                if os.path.exists(font_path):
                    font = processor.ImageFont.truetype(font_path, size)
                    logger.info(f"Font loaded: {font_path}")
                    return font
            except Exception as e:
                logger.warning(f"Font {font_path} failed to load: {e}")
        
        # Use default font as final fallback
        try:
            font = processor.ImageFont.load_default()
            logger.info("Using default font")
            return font
        except Exception as e:
            logger.error(f"Even default font failed: {e}")
            return None
    
    def get_template(self):
        """Get template image with fallbacks"""
        for template_path in self.template_paths:
            try:
                if os.path.exists(template_path):
                    template = processor.safe_image_open(template_path)
                    if template:
                        logger.info(f"Template loaded: {template_path}")
                        return template
            except Exception as e:
                logger.warning(f"Template {template_path} failed to load: {e}")
        
        # Create a basic template if none found
        try:
            template = processor.Image.new('RGB', (800, 1200), color='white')
            logger.info("Created basic template")
            return template
        except Exception as e:
            logger.error(f"Failed to create basic template: {e}")
            return None

config = Config()

def safe_draw_centered(draw, text, y, font, image_width):
    """Safely draw centered text with error handling"""
    try:
        if not font or not text:
            return
        
        # Try getbbox first (newer Pillow versions)
        try:
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            # Fallback for older Pillow versions
            try:
                text_width = font.getsize(text)[0]
            except:
                # Final fallback - estimate width
                text_width = len(text) * 12
        
        x = max(0, (image_width - text_width) // 2)
        draw.text((x, y), text, font=font, fill=config.font_color)
        logger.debug(f"Drew text: {text[:20]}... at position ({x}, {y})")
        
    except Exception as e:
        logger.error(f"Failed to draw text '{text}': {e}")

def create_error_display(error_msg, details=None):
    """Create user-friendly error display"""
    st.error(f"❌ {error_msg}")
    if details:
        with st.expander("Technical Details"):
            st.code(details)
    
    # Provide helpful suggestions
    st.info("💡 Troubleshooting suggestions:")
    st.write("• Check if all required files exist in the correct directories")
    st.write("• Ensure the uploaded image contains a visible QR code")
    st.write("• Try uploading a different image format (PNG, JPG)")
    st.write("• Check the application logs for detailed error information")

# Main application
def main():
    """Main application with comprehensive error handling"""
    try:
        st.title("🎫 Qiddiya Work Access Pass Generator")
        st.markdown("---")
        
        # Input validation decorator
        def validate_inputs():
            errors = []
            if not name.strip():
                errors.append("Full Name is required")
            if not company.strip():
                errors.append("Company Name is required")
            if not id_number.strip():
                errors.append("ID Number is required")
            if expiry_date <= issue_date:
                errors.append("Expiry date must be after issue date")
            return errors
        
        # Create input columns with error handling
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *", key="name", help="Enter your full name").upper().strip()
                company = st.text_input("Company Name *", key="company", help="Enter company name").upper().strip()
                id_number = st.text_input("ID Number *", key="id_number", help="Enter your ID number").strip()
            
            with col2:
                issue_date = st.date_input("Issue Date", key="issue_date", help="Select issue date")
                expiry_date = st.date_input(
                    "Expiry Date", 
                    min_value=issue_date, 
                    key="expiry_date",
                    help="Select expiry date (must be after issue date)"
                )
        
        except Exception as e:
            create_error_display("Failed to create input fields", str(e))
            return
        
        # File uploader with enhanced error handling
        st.markdown("### 📤 Upload Pass Image")
        uploaded_photo = st.file_uploader(
            "Upload Existing Pass (to extract QR code)",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload an image containing a QR code to extract"
        )
        
        if uploaded_photo is not None:
            try:
                # Validate file
                if uploaded_photo.size > 10 * 1024 * 1024:  # 10MB limit
                    st.error("File too large. Please upload an image smaller than 10MB.")
                    return
                
                # Display uploaded image
                st.image(uploaded_photo, caption="Uploaded Image", use_container_width=True)
                
                # Process image
                with st.spinner("🔍 Processing image and detecting QR code..."):
                    photo = processor.safe_image_open(uploaded_photo)
                    
                    if not photo:
                        create_error_display("Failed to open uploaded image", "The image format may not be supported or the file may be corrupted.")
                        return
                    
                    # Detect QR code
                    qr_image = processor.detect_qr_code(photo)
                    
                    if qr_image:
                        st.success("✅ QR code detected successfully!")
                        
                        # Resize QR code
                        try:
                            qr_image = qr_image.resize(config.qr_size, processor.Image.Resampling.LANCZOS)
                        except AttributeError:
                            # Fallback for older Pillow versions
                            qr_image = qr_image.resize(config.qr_size, processor.Image.ANTIALIAS)
                        
                        # Show QR preview
                        col_qr1, col_qr2, col_qr3 = st.columns([1, 2, 1])
                        with col_qr2:
                            st.image(qr_image, caption="Extracted QR Code")
                        
                        # Validate all inputs
                        validation_errors = validate_inputs()
                        if validation_errors:
                            st.error("Please fix the following errors:")
                            for error in validation_errors:
                                st.write(f"• {error}")
                            return
                        
                        # Generate pass button
                        if st.button("🎫 Generate Access Pass", type="primary"):
                            with st.spinner("🔄 Generating your access pass..."):
                                try:
                                    # Load template
                                    template = config.get_template()
                                    if not template:
                                        create_error_display("Template not found", "Could not load the pass template. Please ensure template.png exists.")
                                        return
                                    
                                    # Setup drawing
                                    draw = processor.ImageDraw.Draw(template)
                                    font = config.get_font()
                                    
                                    if not font:
                                        st.warning("⚠️ Font not found, using basic text rendering")
                                    
                                    # Format dates safely
                                    try:
                                        formatted_issue_date = issue_date.strftime("%d/%m/%Y")
                                        formatted_expiry_date = expiry_date.strftime("%d/%m/%Y")
                                    except Exception as e:
                                        logger.error(f"Date formatting error: {e}")
                                        formatted_issue_date = str(issue_date)
                                        formatted_expiry_date = str(expiry_date)
                                    
                                    # Draw content with error handling
                                    try:
                                        # Issue date
                                        safe_draw_centered(draw, formatted_issue_date, 240, font, template.width)
                                        
                                        # Paste QR code
                                        qr_pos_x = (template.width - config.qr_size[0]) // 2
                                        qr_pos_y = 471
                                        template.paste(qr_image, (qr_pos_x, qr_pos_y))
                                        
                                        # Text below QR code
                                        y_pos = qr_pos_y + config.qr_size[1] + 60
                                        safe_draw_centered(draw, f"EX {formatted_expiry_date}", y_pos, font, template.width)
                                        y_pos += 30
                                        safe_draw_centered(draw, name, y_pos, font, template.width)
                                        y_pos += 30
                                        safe_draw_centered(draw, company, y_pos, font, template.width)
                                        y_pos += 30
                                        safe_draw_centered(draw, id_number, y_pos, font, template.width)
                                        
                                    except Exception as e:
                                        logger.error(f"Error drawing content: {e}")
                                        st.error(f"Error generating pass content: {e}")
                                        return
                                    
                                    # Show preview
                                    st.success("✅ Access pass generated successfully!")
                                    st.image(template, caption="🎫 Your Access Pass Preview", use_container_width=True)
                                    
                                    # Prepare download
                                    try:
                                        buf = io.BytesIO()
                                        template.save(buf, format="PNG", optimize=True)
                                        buf.seek(0)
                                        
                                        # Safe filename
                                        safe_name = "".join(c for c in name.lower() if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
                                        filename = f"qiddiya_pass_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                        
                                        st.download_button(
                                            "📥 Download Access Pass",
                                            data=buf.getvalue(),
                                            file_name=filename,
                                            mime="image/png",
                                            help="Click to download your generated access pass"
                                        )
                                        
                                        logger.info(f"Pass generated successfully for {name}")
                                        
                                    except Exception as e:
                                        create_error_display("Failed to prepare download", str(e))
                                
                                except Exception as e:
                                    create_error_display("Failed to generate pass", str(e))
                                    logger.error(f"Pass generation failed: {traceback.format_exc()}")
                    
                    else:
                        st.error("❌ No QR code detected in the uploaded image.")
                        st.markdown("**Possible solutions:**")
                        st.write("• Ensure the image contains a clear, visible QR code")
                        st.write("• Try uploading a higher quality image")
                        st.write("• Make sure the QR code is not obscured or distorted")
                        st.write("• Try a different image format")
                        
                        # Manual crop option
                        st.info("💡 **Alternative:** If you know where the QR code is, you can manually crop it first")
            
            except Exception as e:
                create_error_display("Error processing uploaded image", str(e))
                logger.error(f"Image processing error: {traceback.format_exc()}")
        
        else:
            st.info("📋 Please upload an image containing a QR code to get started.")
        
        # Footer with system info
        st.markdown("---")
        with st.expander("🔧 System Information"):
            st.write(f"**Available QR Libraries:** {', '.join(processor.qr_libs)}")
            st.write(f"**PIL Available:** {'✅' if processor.pil_available else '❌'}")
            st.write(f"**Supported Formats:** {', '.join(['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])}")
    
    except Exception as e:
        st.error("🚨 Critical application error occurred!")
        logger.critical(f"Critical error: {traceback.format_exc()}")
        
        with st.expander("Error Details"):
            st.code(str(e))
            st.code(traceback.format_exc())
        
        st.info("Please check the logs and try refreshing the page.")

# Error boundary
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("🚨 Application failed to start!")
        st.code(str(e))
        logger.critical(f"App startup failed: {traceback.format_exc()}")