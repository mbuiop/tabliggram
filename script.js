// Advanced Invoice Generator with High-Quality PNG Export
class AdvancedInvoiceGenerator {
    constructor() {
        this.currentTemplate = 'modern';
        this.currentColor = '#4361ee';
        this.currentDateFormat = 'jalali';
        this.fontSize = 16;
        this.items = [
            {
                id: 1,
                description: 'گوشی موبایل سامسونگ S24 256GB',
                quantity: 1,
                unitPrice: 45000000,
                discount: 5
            },
            {
                id: 2,
                description: 'پاوربانک شیائومی ۲۰۰۰۰ میلی آمپر',
                quantity: 1,
                unitPrice: 2500000,
                discount: 0
            }
        ];
        
        this.init();
    }
    
    init() {
        this.loadPersianFont();
        this.setupEventListeners();
        this.renderInvoice();
        this.renderItemsList();
        this.calculateTotals();
    }
    
    async loadPersianFont() {
        // Load a high-quality Persian font for canvas rendering
        const font = new FontFace('Vazirmatn', 'url(https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/fonts/webfonts/Vazirmatn-Regular.woff2)');
        
        try {
            await font.load();
            document.fonts.add(font);
            console.log('Persian font loaded successfully');
        } catch (error) {
            console.error('Failed to load font:', error);
        }
    }
    
    setupEventListeners() {
        // Color picker
        document.getElementById('colorPicker').addEventListener('input', (e) => {
            this.currentColor = e.target.value;
            this.renderInvoice();
        });
        
        // Template buttons
        document.querySelectorAll('.template-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.template-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.currentTemplate = btn.dataset.template;
                this.currentColor = btn.dataset.color;
                document.getElementById('colorPicker').value = this.currentColor;
                this.renderInvoice();
            });
        });
        
        // Date format
        document.querySelectorAll('input[name="dateFormat"]').forEach(radio => {
            radio.addEventListener('change', () => {
                this.currentDateFormat = radio.value;
                this.renderInvoice();
            });
        });
        
        // Font size
        document.getElementById('fontSize').addEventListener('input', (e) => {
            this.fontSize = parseInt(e.target.value);
            this.renderInvoice();
        });
        
        // Download button with advanced rendering
        document.getElementById('downloadBtn').addEventListener('click', () => {
            this.downloadHighQualityInvoice();
        });
        
        // Input fields
        ['sellerName', 'sellerPhone', 'sellerAddress', 'buyerName', 'buyerPhone', 'invoiceNumber']
            .forEach(id => {
                document.getElementById(id).addEventListener('input', () => this.renderInvoice());
            });
        
        // Add item
        document.getElementById('addItemBtn').addEventListener('click', () => this.addItem());
    }
    
    renderInvoice() {
        const invoiceData = this.getInvoiceData();
        const html = this.generateInvoiceHTML(invoiceData);
        document.getElementById('invoiceContainer').innerHTML = html;
        
        // Apply custom styles for the template
        this.applyTemplateStyles();
    }
    
    getInvoiceData() {
        return {
            sellerName: document.getElementById('sellerName').value,
            sellerPhone: document.getElementById('sellerPhone').value,
            sellerAddress: document.getElementById('sellerAddress').value,
            buyerName: document.getElementById('buyerName').value,
            buyerPhone: document.getElementById('buyerPhone').value,
            invoiceNumber: document.getElementById('invoiceNumber').value,
            date: this.getFormattedDate(),
            items: this.items,
            totals: this.calculateInvoiceTotals()
        };
    }
    
    getFormattedDate() {
        const now = new Date();
        const jalali = this.getJalaliDate(now);
        const gregorian = this.getGregorianDate(now);
        
        switch(this.currentDateFormat) {
            case 'jalali': return jalali;
            case 'gregorian': return gregorian;
            case 'both': return `${jalali} | ${gregorian}`;
            default: return jalali;
        }
    }
    
    getJalaliDate(date) {
        // Using a simple conversion - for production use jalali-js library
        const jalaliMonths = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 
                             'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'];
        const jalaliYear = 1403;
        const month = jalaliMonths[date.getMonth()];
        const day = date.getDate();
        return `${jalaliYear}/${month}/${day}`;
    }
    
    getGregorianDate(date) {
        return date.toLocaleDateString('en-CA');
    }
    
    calculateInvoiceTotals() {
        let subtotal = 0;
        let totalDiscount = 0;
        
        this.items.forEach(item => {
            const itemTotal = item.quantity * item.unitPrice;
            subtotal += itemTotal;
            totalDiscount += itemTotal * (item.discount / 100);
        });
        
        const tax = (subtotal - totalDiscount) * 0.09;
        const total = subtotal - totalDiscount + tax;
        
        return {
            subtotal,
            discount: totalDiscount,
            tax,
            total,
            totalInWords: this.numberToPersianWords(total)
        };
    }
    
    generateInvoiceHTML(data) {
        return `
            <div class="invoice-preview ${this.currentTemplate}" id="invoiceForExport">
                ${this.generateHeaderHTML(data)}
                ${this.generatePartiesHTML(data)}
                ${this.generateItemsTableHTML(data)}
                ${this.generateSummaryHTML(data)}
                ${this.generateFooterHTML(data)}
            </div>
        `;
    }
    
    generateHeaderHTML(data) {
        return `
            <div class="invoice-header" style="border-color: ${this.currentColor}">
                <div class="header-left">
                    <h1 class="invoice-title" style="color: ${this.currentColor}">فاکتور فروش</h1>
                    <div class="invoice-meta">
                        <div class="invoice-number">شماره: <strong>${data.invoiceNumber}</strong></div>
                        <div class="invoice-date">تاریخ: <strong>${data.date}</strong></div>
                    </div>
                </div>
                <div class="header-right">
                    <div class="watermark">${data.sellerName}</div>
                </div>
            </div>
        `;
    }
    
    generatePartiesHTML(data) {
        return `
            <div class="parties-section">
                <div class="party-card seller">
                    <h3 style="color: ${this.currentColor}">فروشنده</h3>
                    <div class="party-details">
                        <p><i class="fas fa-store"></i> ${data.sellerName}</p>
                        <p><i class="fas fa-phone"></i> ${data.sellerPhone}</p>
                        <p><i class="fas fa-map-marker-alt"></i> ${data.sellerAddress}</p>
                    </div>
                </div>
                <div class="party-card buyer">
                    <h3 style="color: ${this.currentColor}">خریدار</h3>
                    <div class="party-details">
                        <p><i class="fas fa-user"></i> ${data.buyerName}</p>
                        <p><i class="fas fa-mobile-alt"></i> ${data.buyerPhone}</p>
                    </div>
                </div>
            </div>
        `;
    }
    
    generateItemsTableHTML(data) {
        return `
            <div class="table-container">
                <table class="items-table">
                    <thead>
                        <tr style="background: ${this.currentColor}; color: white;">
                            <th>ردیف</th>
                            <th>شرح کالا / خدمت</th>
                            <th>تعداد</th>
                            <th>قیمت واحد (ریال)</th>
                            <th>تخفیف %</th>
                            <th>مجموع (ریال)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.items.map((item, index) => {
                            const total = item.quantity * item.unitPrice;
                            const afterDiscount = total * (1 - item.discount/100);
                            return `
                                <tr class="${index % 2 === 0 ? 'even' : 'odd'}">
                                    <td>${index + 1}</td>
                                    <td>${item.description}</td>
                                    <td>${item.quantity}</td>
                                    <td>${this.formatNumber(item.unitPrice)}</td>
                                    <td>${item.discount}%</td>
                                    <td>${this.formatNumber(afterDiscount)}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
        `;
    }
    
    generateSummaryHTML(data) {
        return `
            <div class="summary-section">
                <div class="summary-grid">
                    <div class="summary-item">
                        <span>جمع کل:</span>
                        <span>${this.formatCurrency(data.totals.subtotal)}</span>
                    </div>
                    <div class="summary-item">
                        <span>تخفیف:</span>
                        <span>${this.formatCurrency(data.totals.discount)}</span>
                    </div>
                    <div class="summary-item">
                        <span>مالیات ارزش افزوده (۹٪):</span>
                        <span>${this.formatCurrency(data.totals.tax)}</span>
                    </div>
                    <div class="summary-item total" style="border-color: ${this.currentColor}">
                        <span>مبلغ قابل پرداخت:</span>
                        <span style="color: ${this.currentColor}">${this.formatCurrency(data.totals.total)}</span>
                    </div>
                </div>
                <div class="amount-in-words">
                    <strong>مبلغ به حروف:</strong> ${data.totals.totalInWords}
                </div>
            </div>
        `;
    }
    
    generateFooterHTML() {
        return `
            <div class="footer-section">
                <div class="notes">
                    <p><strong>توضیحات:</strong> کالاها ظرف ۳ روز کاری آماده تحویل می‌باشد. گارانتی ۱۸ ماهه</p>
                    <p><strong>روش پرداخت:</strong> نقدی</p>
                </div>
                <div class="signatures">
                    <div class="signature-box">
                        <p>امضای خریدار</p>
                        <div class="signature-line"></div>
                    </div>
                    <div class="signature-box">
                        <p>امضا و مهر فروشنده</p>
                        <div class="signature-line"></div>
                    </div>
                </div>
                <div class="footer-note">
                    <p>این فاکتور به صورت الکترونیکی صادر شده و نیاز به مهر و امضای فیزیکی ندارد</p>
                </div>
            </div>
        `;
    }
    
    applyTemplateStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .invoice-preview {
                font-family: 'Vazirmatn', sans-serif;
                font-size: ${this.fontSize}px;
                line-height: 1.8;
                padding: 40px;
                background: white;
                color: #333;
                border-radius: 8px;
                box-shadow: 0 0 30px rgba(0,0,0,0.1);
            }
            
            .invoice-preview.modern {
                border: 2px solid ${this.currentColor};
            }
            
            .invoice-preview.classic {
                background: #fefefe;
                border: 1px solid #ddd;
                font-family: 'Vazirmatn', serif;
            }
            
            .invoice-preview.minimal {
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
            }
            
            .invoice-header {
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                padding-bottom: 30px;
                margin-bottom: 30px;
                border-bottom: 3px solid;
            }
            
            .invoice-title {
                font-size: ${this.fontSize * 2}px;
                font-weight: 900;
                margin-bottom: 10px;
            }
            
            .watermark {
                opacity: 0.1;
                font-size: 60px;
                transform: rotate(-15deg);
                position: absolute;
                right: 100px;
                top: 150px;
            }
            
            .parties-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 40px;
                margin-bottom: 40px;
            }
            
            .party-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                border-right: 5px solid ${this.currentColor};
            }
            
            .party-card h3 {
                font-size: ${this.fontSize * 1.2}px;
                margin-bottom: 15px;
            }
            
            .table-container {
                margin: 40px 0;
                overflow-x: auto;
            }
            
            .items-table {
                width: 100%;
                border-collapse: collapse;
                border-spacing: 0;
            }
            
            .items-table th {
                padding: 15px;
                text-align: right;
                font-weight: 700;
                border-bottom: 2px solid ${this.currentColor};
            }
            
            .items-table td {
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }
            
            .items-table tr.even {
                background: #fafafa;
            }
            
            .items-table tr:hover {
                background: #f0f7ff;
            }
            
            .summary-section {
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
                padding: 30px;
                border-radius: 10px;
                margin: 40px 0;
                border: 2px dashed ${this.currentColor}40;
            }
            
            .summary-grid {
                display: grid;
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .summary-item {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }
            
            .summary-item.total {
                border-bottom: 3px solid;
                font-size: ${this.fontSize * 1.3}px;
                font-weight: 900;
                padding: 15px 0;
            }
            
            .amount-in-words {
                background: white;
                padding: 15px;
                border-radius: 8px;
                border-right: 4px solid ${this.currentColor};
                margin-top: 20px;
            }
            
            .footer-section {
                margin-top: 50px;
                padding-top: 30px;
                border-top: 2px solid #eee;
            }
            
            .signatures {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 50px;
                margin: 40px 0;
            }
            
            .signature-box {
                text-align: center;
            }
            
            .signature-line {
                height: 2px;
                background: #333;
                margin-top: 60px;
                position: relative;
            }
            
            .signature-line::after {
                content: '';
                position: absolute;
                width: 100%;
                height: 20px;
                border-bottom: 1px dashed #333;
                bottom: -30px;
            }
            
            .footer-note {
                text-align: center;
                color: #666;
                font-style: italic;
                margin-top: 30px;
                padding: 15px;
                background: #f8f9fa;
                border-radius: 8px;
            }
        `;
        
        const oldStyle = document.querySelector('#invoice-styles');
        if (oldStyle) oldStyle.remove();
        style.id = 'invoice-styles';
        document.head.appendChild(style);
    }
    
    async downloadHighQualityInvoice() {
        const element = document.getElementById('invoiceForExport');
        const btn = document.getElementById('downloadBtn');
        
        // Save original button state
        const originalHTML = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال ایجاد با کیفیت بالا...';
        btn.disabled = true;
        
        // Wait for fonts to load
        await document.fonts.ready;
        
        // Use html2canvas with advanced options for better Persian text rendering
        const canvas = await html2canvas(element, {
            scale: 4, // Very high resolution
            useCORS: true,
            allowTaint: true,
            backgroundColor: '#ffffff',
            logging: false,
            onclone: function(clonedDoc) {
                // Ensure fonts are loaded in the cloned document
                clonedDoc.fonts = document.fonts;
            },
            letterRendering: true,
            fontEmbedCSS: true
        });
        
        // Create a new canvas for final touch-ups
        const finalCanvas = document.createElement('canvas');
        const ctx = finalCanvas.getContext('2d');
        
        // Set final dimensions (A4 ratio)
        finalCanvas.width = 2480; // 8.27 inches * 300 DPI
        finalCanvas.height = 3508; // 11.69 inches * 300 DPI
        
        // Fill with white background
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, finalCanvas.width, finalCanvas.height);
        
        // Calculate scaling to fit A4
        const scale = Math.min(
            (finalCanvas.width - 200) / canvas.width,
            (finalCanvas.height - 200) / canvas.height
        );
        
        const x = (finalCanvas.width - canvas.width * scale) / 2;
        const y = (finalCanvas.height - canvas.height * scale) / 2;
        
        // Draw the captured canvas with anti-aliasing
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(canvas, x, y, canvas.width * scale, canvas.height * scale);
        
        // Add watermark/logo
        this.addWatermark(ctx, finalCanvas);
        
        // Create download link
        const link = document.createElement('a');
        link.download = `فاکتور-${document.getElementById('invoiceNumber').value}-HQ.png`;
        link.href = finalCanvas.toDataURL('image/png', 1.0); // Maximum quality
        link.click();
        
        // Restore button
        btn.innerHTML = originalHTML;
        btn.disabled = false;
        
        // Show success message
        this.showNotification('فاکتور با کیفیت بالا دانلود شد!', 'success');
    }
    
    addWatermark(ctx, canvas) {
        ctx.save();
        ctx.globalAlpha = 0.03;
        ctx.font = 'bold 120px Vazirmatn';
        ctx.fillStyle = this.currentColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 4);
        ctx.fillText('فاکتورساز حرفه‌ای', 0, 0);
        ctx.restore();
    }
    
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
            ${message}
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    // Helper methods for items list
    renderItemsList() {
        const container = document.getElementById('itemsList');
        container.innerHTML = '';
        
        this.items.forEach((item, index) => {
            const itemElement = document.createElement('div');
            itemElement.className = 'item-row';
            itemElement.innerHTML = `
                <input type="text" class="item-desc" value="${item.description}" 
                       data-index="${index}" data-field="description">
                <input type="number" class="item-qty" value="${item.quantity}" min="1"
                       data-index="${index}" data-field="quantity">
                <input type="number" class="item-price" value="${item.unitPrice}" 
                       data-index="${index}" data-field="unitPrice">
                <input type="number" class="item-discount" value="${item.discount}" min="0" max="100"
                       data-index="${index}" data-field="discount">
                <span class="item-total">${this.formatCurrency(item.quantity * item.unitPrice * (1 - item.discount/100))}</span>
                <button class="remove-item" data-index="${index}" ${this.items.length <= 1 ? 'disabled' : ''}>×</button>
            `;
            
            container.appendChild(itemElement);
        });
        
        this.setupItemEvents();
    }
    
    setupItemEvents() {
        document.querySelectorAll('.item-desc, .item-qty, .item-price, .item-discount').forEach(input => {
            input.addEventListener('input', (e) => {
                const index = parseInt(e.target.dataset.index);
                const field = e.target.dataset.field;
                const value = e.target.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
                
                this.items[index][field] = value;
                this.renderInvoice();
                this.calculateTotals();
                
                // Update total for this row
                if (field !== 'description') {
                    const item = this.items[index];
                    const total = item.quantity * item.unitPrice * (1 - item.discount/100);
                    e.target.closest('.item-row').querySelector('.item-total').textContent = this.formatCurrency(total);
                }
            });
        });
        
        document.querySelectorAll('.remove-item').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const index = parseInt(e.target.dataset.index);
                if (this.items.length > 1) {
                    this.items.splice(index, 1);
                    this.renderItemsList();
                    this.renderInvoice();
                    this.calculateTotals();
                }
            });
        });
    }
    
    addItem() {
        const newItem = {
            id: this.items.length + 1,
            description: 'کالا/خدمت جدید',
            quantity: 1,
            unitPrice: 1000000,
            discount: 0
        };
        this.items.push(newItem);
        this.renderItemsList();
        this.renderInvoice();
        this.calculateTotals();
    }
    
    calculateTotals() {
        const totals = this.calculateInvoiceTotals();
        
        document.getElementById('subtotal').textContent = this.formatCurrency(totals.subtotal);
        document.getElementById('discountAmount').textContent = this.formatCurrency(totals.discount);
        document.getElementById('taxAmount').textContent = this.formatCurrency(totals.tax);
        document.getElementById('totalAmount').textContent = this.formatCurrency(totals.total);
    }
    
    // Utility methods
    formatNumber(num) {
        return new Intl.NumberFormat('fa-IR').format(num);
    }
    
    formatCurrency(num) {
        return this.formatNumber(num) + ' ریال';
    }
    
    numberToPersianWords(num) {
        // Simplified version - for production use a proper library like persian-tools
        const units = ['', 'هزار', 'میلیون', 'میلیارد'];
        let result = '';
        
        if (num >= 1000000000) {
            const billions = Math.floor(num / 1000000000);
            result += this.formatNumber(billions) + ' میلیارد و ';
            num %= 1000000000;
        }
        
        if (num >= 1000000) {
            const millions = Math.floor(num / 1000000);
            result += this.formatNumber(millions) + ' میلیون و ';
            num %= 1000000;
        }
        
        if (num >= 1000) {
            const thousands = Math.floor(num / 1000);
            result += this.formatNumber(thousands) + ' هزار و ';
            num %= 1000;
        }
        
        if (num > 0) {
            result += this.formatNumber(num);
        }
        
        return result + ' تومان';
    }
}

// Initialize the advanced invoice generator when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.invoiceApp = new AdvancedInvoiceGenerator();
});
