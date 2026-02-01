// فاکتورساز پیشرفته - نسخه نهایی
class InvoiceGeneratorPro {
    constructor() {
        this.config = {
            template: 'modern',
            color: '#4361ee',
            dateFormat: 'jalali',
            fontSize: 16
        };
        
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
        this.loadFonts();
        this.setupEventListeners();
        this.renderInvoice();
        this.renderItems();
        this.updateTotals();
    }
    
    async loadFonts() {
        // بارگذاری فونت فارسی با کیفیت بالا
        const font = new FontFace('Vazirmatn', 
            'url(https://cdn.jsdelivr.net/gh/rastikerdar/vazirmatn@v33.003/fonts/webfonts/Vazirmatn[wght].woff2)',
            { weight: '300 900' }
        );
        
        try {
            await font.load();
            document.fonts.add(font);
        } catch (error) {
            console.warn('خطا در بارگذاری فونت:', error);
        }
    }
    
    setupEventListeners() {
        // انتخاب قالب
        document.querySelectorAll('.template-card').forEach(card => {
            card.addEventListener('click', () => {
                document.querySelectorAll('.template-card').forEach(c => c.classList.remove('active'));
                card.classList.add('active');
                this.config.template = card.dataset.template;
                this.config.color = card.dataset.color;
                document.getElementById('customColor').value = this.config.color;
                this.updateColorName();
                this.renderInvoice();
            });
        });
        
        // انتخاب رنگ از پالت
        document.querySelectorAll('.color-option').forEach(option => {
            option.addEventListener('click', () => {
                document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
                option.classList.add('active');
                this.config.color = option.dataset.color;
                document.getElementById('customColor').value = this.config.color;
                document.getElementById('currentColorName').textContent = option.dataset.name;
                this.renderInvoice();
            });
        });
        
        // انتخاب رنگ دلخواه
        document.getElementById('customColor').addEventListener('input', (e) => {
            this.config.color = e.target.value;
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            document.getElementById('currentColorName').textContent = 'رنگ دلخواه';
            this.renderInvoice();
        });
        
        // فرمت تاریخ
        document.querySelectorAll('input[name="dateFormat"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.config.dateFormat = e.target.value;
                this.renderInvoice();
            });
        });
        
        // اندازه فونت
        document.getElementById('fontSize').addEventListener('input', (e) => {
            this.config.fontSize = parseInt(e.target.value);
            document.getElementById('currentFontSize').textContent = `${this.config.fontSize}px`;
            this.renderInvoice();
        });
        
        // کپی شماره کارت
        document.querySelector('.copy-btn').addEventListener('click', () => {
            const text = document.querySelector('.copy-btn').dataset.text;
            navigator.clipboard.writeText(text).then(() => {
                this.showNotification('شماره کارت کپی شد!', 'success');
            });
        });
        
        // فیلدهای ورودی
        ['sellerName', 'sellerPhone', 'sellerAddress', 'buyerName', 'buyerPhone', 'invoiceNumber']
            .forEach(id => {
                document.getElementById(id).addEventListener('input', () => this.renderInvoice());
            });
        
        // افزودن آیتم جدید
        document.getElementById('addItemBtn').addEventListener('click', () => this.addNewItem());
        
        // دانلود فاکتور
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadInvoice());
        
        // بازنشانی
        document.getElementById('resetBtn').addEventListener('click', () => this.resetToDefaults());
    }
    
    updateColorName() {
        const colorOptions = document.querySelectorAll('.color-option');
        let found = false;
        
        colorOptions.forEach(option => {
            if (option.dataset.color === this.config.color) {
                document.getElementById('currentColorName').textContent = option.dataset.name;
                option.classList.add('active');
                found = true;
            }
        });
        
        if (!found) {
            document.getElementById('currentColorName').textContent = 'رنگ دلخواه';
        }
    }
    
    renderInvoice() {
        const data = this.getInvoiceData();
        const html = this.generateInvoiceHTML(data);
        document.getElementById('invoicePreview').innerHTML = html;
    }
    
    getInvoiceData() {
        return {
            seller: {
                name: document.getElementById('sellerName').value || 'فروشگاه موبایل پارسیان',
                phone: document.getElementById('sellerPhone').value || '021-12345678',
                address: document.getElementById('sellerAddress').value || 'تهران، خیابان ولیعصر، پلاک ۲۴۰'
            },
            buyer: {
                name: document.getElementById('buyerName').value || 'سعید کنانی',
                phone: document.getElementById('buyerPhone').value || '09137657870'
            },
            invoiceNumber: document.getElementById('invoiceNumber').value || 'INV-1403-001',
            date: this.getFormattedDate(),
            items: this.items,
            totals: this.calculateTotals(),
            config: this.config
        };
    }
    
    getFormattedDate() {
        const now = new Date();
        const jalali = this.toJalali(now);
        const gregorian = this.toGregorian(now);
        
        switch(this.config.dateFormat) {
            case 'jalali': return jalali;
            case 'gregorian': return gregorian;
            case 'both': return `${jalali} | ${gregorian}`;
            default: return jalali;
        }
    }
    
    toJalali(date) {
        // تبدیل ساده تاریخ - برای استفاده واقعی از کتابخانه jalali-js استفاده کنید
        const jalaliMonths = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 
                             'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'];
        const jalaliYear = 1403;
        const month = jalaliMonths[date.getMonth()];
        const day = date.getDate();
        return `${jalaliYear}/${month}/${day}`;
    }
    
    toGregorian(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}/${month}/${day}`;
    }
    
    calculateTotals() {
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
        const { seller, buyer, invoiceNumber, date, items, totals, config } = data;
        
        return `
            <div class="invoice" style="font-size: ${config.fontSize}px; --primary-color: ${config.color}">
                <!-- Header -->
                <div class="invoice-header" style="border-color: ${config.color}">
                    <div class="header-left">
                        <h1 style="color: ${config.color}">فاکتور فروش</h1>
                        <div class="invoice-info">
                            <div class="info-row">
                                <span>شماره:</span>
                                <strong>${invoiceNumber}</strong>
                            </div>
                            <div class="info-row">
                                <span>تاریخ:</span>
                                <strong>${date}</strong>
                            </div>
                        </div>
                    </div>
                    <div class="header-right">
                        <div class="watermark">${seller.name}</div>
                    </div>
                </div>
                
                <!-- Parties -->
                <div class="parties">
                    <div class="party seller">
                        <h3 style="color: ${config.color}">فروشنده</h3>
                        <div class="party-details">
                            <p><i class="fas fa-store"></i> ${seller.name}</p>
                            <p><i class="fas fa-phone"></i> ${seller.phone}</p>
                            <p><i class="fas fa-map-marker-alt"></i> ${seller.address}</p>
                        </div>
                    </div>
                    <div class="party buyer">
                        <h3 style="color: ${config.color}">خریدار</h3>
                        <div class="party-details">
                            <p><i class="fas fa-user"></i> ${buyer.name}</p>
                            <p><i class="fas fa-mobile-alt"></i> ${buyer.phone}</p>
                        </div>
                    </div>
                </div>
                
                <!-- Items Table -->
                <table class="items-table">
                    <thead style="background: ${config.color}">
                        <tr>
                            <th>ردیف</th>
                            <th>شرح کالا / خدمت</th>
                            <th>تعداد</th>
                            <th>قیمت واحد (ریال)</th>
                            <th>تخفیف %</th>
                            <th>مجموع (ریال)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${items.map((item, index) => {
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
                
                <!-- Summary -->
                <div class="summary" style="border-color: ${config.color}">
                    <div class="summary-row">
                        <span>جمع کل:</span>
                        <span>${this.formatCurrency(totals.subtotal)}</span>
                    </div>
                    <div class="summary-row">
                        <span>تخفیف:</span>
                        <span>${this.formatCurrency(totals.discount)}</span>
                    </div>
                    <div class="summary-row">
                        <span>مالیات بر ارزش افزوده (۹٪):</span>
                        <span>${this.formatCurrency(totals.tax)}</span>
                    </div>
                    <div class="summary-row total">
                        <span>مبلغ قابل پرداخت:</span>
                        <span style="color: ${config.color}">${this.formatCurrency(totals.total)}</span>
                    </div>
                    <div class="amount-in-words">
                        <strong>مبلغ به حروف:</strong> ${totals.totalInWords}
                    </div>
                </div>
                
                <!-- Footer -->
                <div class="invoice-footer">
                    <div class="notes">
                        <p><strong>توضیحات:</strong> کالاها ظرف ۳ روز کاری آماده تحویل می‌باشد. گارانتی ۱۸ ماهه</p>
                        <p><strong>روش پرداخت:</strong> نقدی</p>
                    </div>
                    <div class="signatures">
                        <div class="signature">
                            <p>امضای خریدار</p>
                            <div class="line"></div>
                        </div>
                        <div class="signature">
                            <p>امضا و مهر فروشنده</p>
                            <div class="line"></div>
                        </div>
                    </div>
                    <div class="footer-note">
                        این فاکتور به صورت الکترونیکی صادر شده و نیاز به مهر و امضای فیزیکی ندارد
                    </div>
                </div>
            </div>
            
            <style>
                .invoice {
                    font-family: 'Vazirmatn', sans-serif;
                    direction: rtl;
                    color: #333;
                    line-height: 1.8;
                }
                
                .invoice-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: flex-start;
                    padding-bottom: 30px;
                    margin-bottom: 40px;
                    border-bottom: 3px solid;
                }
                
                .invoice-header h1 {
                    font-size: 2.5em;
                    font-weight: 900;
                    margin-bottom: 10px;
                }
                
                .invoice-info {
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                }
                
                .info-row {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 5px;
                }
                
                .watermark {
                    opacity: 0.1;
                    font-size: 4em;
                    font-weight: 900;
                    transform: rotate(-15deg);
                }
                
                .parties {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 40px;
                    margin-bottom: 40px;
                }
                
                .party {
                    background: #f8f9fa;
                    padding: 25px;
                    border-radius: 12px;
                    border-right: 5px solid;
                }
                
                .party h3 {
                    font-size: 1.4em;
                    margin-bottom: 20px;
                }
                
                .party-details p {
                    margin-bottom: 10px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                
                .items-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 40px 0;
                }
                
                .items-table th {
                    color: white;
                    padding: 15px;
                    text-align: right;
                    font-weight: 700;
                }
                
                .items-table td {
                    padding: 12px 15px;
                    border-bottom: 1px solid #eee;
                }
                
                .items-table tr.even {
                    background: #fafafa;
                }
                
                .summary {
                    background: #f8f9fa;
                    padding: 30px;
                    border-radius: 12px;
                    margin: 40px 0;
                    border: 2px dashed;
                }
                
                .summary-row {
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px dashed #ddd;
                }
                
                .summary-row.total {
                    border-bottom: 3px solid;
                    font-size: 1.4em;
                    font-weight: 900;
                    margin-top: 10px;
                    padding-top: 20px;
                }
                
                .amount-in-words {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                    border-right: 4px solid ${config.color};
                }
                
                .invoice-footer {
                    margin-top: 50px;
                    padding-top: 30px;
                    border-top: 2px solid #eee;
                }
                
                .signatures {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 60px;
                    margin: 40px 0;
                }
                
                .signature {
                    text-align: center;
                }
                
                .signature .line {
                    height: 2px;
                    background: #333;
                    margin-top: 60px;
                    position: relative;
                }
                
                .signature .line::after {
                    content: '';
                    position: absolute;
                    width: 100%;
                    height: 20px;
                    border-bottom: 1px dashed #666;
                    bottom: -30px;
                }
                
                .footer-note {
                    text-align: center;
                    color: #666;
                    font-style: italic;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                }
            </style>
        `;
    }
    
    renderItems() {
        const container = document.getElementById('itemsContainer');
        container.innerHTML = '';
        
        this.items.forEach((item, index) => {
            const itemElement = document.createElement('div');
            itemElement.className = 'item-row';
            itemElement.innerHTML = `
                <input type="text" class="item-desc" value="${item.description}" 
                       data-index="${index}" data-field="description" placeholder="شرح کالا">
                <input type="number" class="item-qty" value="${item.quantity}" min="1"
                       data-index="${index}" data-field="quantity">
                <input type="number" class="item-price" value="${item.unitPrice}" 
                       data-index="${index}" data-field="unitPrice" placeholder="قیمت">
                <input type="number" class="item-discount" value="${item.discount}" min="0" max="100"
                       data-index="${index}" data-field="discount" placeholder="تخفیف">
                <span class="item-total">${this.formatCurrency(item.quantity * item.unitPrice * (1 - item.discount/100))}</span>
                <button class="remove-item" data-index="${index}" ${this.items.length <= 1 ? 'disabled' : ''}>
                    <i class="fas fa-trash"></i>
                </button>
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
                this.updateTotals();
                
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
                    this.renderItems();
                    this.renderInvoice();
                    this.updateTotals();
                    this.showNotification('آیتم حذف شد', 'success');
                }
            });
        });
    }
    
    addNewItem() {
        const newItem = {
            id: this.items.length + 1,
            description: 'کالا یا خدمات جدید',
            quantity: 1,
            unitPrice: 1000000,
            discount: 0
        };
        
        this.items.push(newItem);
        this.renderItems();
        this.renderInvoice();
        this.updateTotals();
        this.showNotification('آیتم جدید اضافه شد', 'success');
    }
    
    updateTotals() {
        const totals = this.calculateTotals();
        
        document.getElementById('subtotal').textContent = this.formatCurrency(totals.subtotal);
        document.getElementById('discountAmount').textContent = this.formatCurrency(totals.discount);
        document.getElementById('taxAmount').textContent = this.formatCurrency(totals.tax);
        document.getElementById('totalAmount').textContent = this.formatCurrency(totals.total);
        
        // Update invoice count
        const count = parseInt(document.getElementById('invoiceCount').textContent) || 1;
        document.getElementById('invoiceCount').textContent = count + 1;
    }
    
    async downloadInvoice() {
        const btn = document.getElementById('downloadBtn');
        const originalHTML = btn.innerHTML;
        
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال ایجاد...';
        btn.disabled = true;
        
        try {
            // Wait for fonts to load
            await document.fonts.ready;
            
            const element = document.querySelector('.invoice');
            const canvas = await html2canvas(element, {
                scale: 3,
                useCORS: true,
                allowTaint: true,
                backgroundColor: '#ffffff',
                logging: false,
                letterRendering: true
            });
            
            const link = document.createElement('a');
            link.download = `فاکتور-${document.getElementById('invoiceNumber').value || 'جدید'}.png`;
            link.href = canvas.toDataURL('image/png', 1.0);
            link.click();
            
            this.showNotification('فاکتور با موفقیت دانلود شد!', 'success');
        } catch (error) {
            this.showNotification('خطا در ایجاد فاکتور', 'error');
            console.error('Download error:', error);
        } finally {
            btn.innerHTML = originalHTML;
            btn.disabled = false;
        }
    }
    
    resetToDefaults() {
        if (confirm('آیا از بازنشانی همه تنظیمات به حالت اولیه اطمینان دارید؟')) {
            this.config = {
                template: 'modern',
                color: '#4361ee',
                dateFormat: 'jalali',
                fontSize: 16
            };
            
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
            
            // Reset UI
            document.querySelectorAll('.template-card').forEach(c => c.classList.remove('active'));
            document.querySelector('.template-card[data-template="modern"]').classList.add('active');
            
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            document.querySelector('.color-option[data-color="#4361ee"]').classList.add('active');
            
            document.getElementById('customColor').value = '#4361ee';
            document.getElementById('currentColorName').textContent = 'آبی جذاب';
            
            document.querySelector('input[name="dateFormat"][value="jalali"]').checked = true;
            
            document.getElementById('fontSize').value = 16;
            document.getElementById('currentFontSize').textContent = '16px';
            
            // Reset inputs
            document.getElementById('sellerName').value = 'فروشگاه موبایل پارسیان';
            document.getElementById('sellerPhone').value = '021-12345678';
            document.getElementById('sellerAddress').value = 'تهران، خیابان ولیعصر، پلاک ۲۴۰';
            document.getElementById('buyerName').value = 'سعید کنانی';
            document.getElementById('buyerPhone').value = '09137657870';
            document.getElementById('invoiceNumber').value = 'INV-1403-001';
            
            this.renderInvoice();
            this.renderItems();
            this.updateTotals();
            
            this.showNotification('همه تنظیمات بازنشانی شد', 'success');
        }
    }
    
    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        notification.textContent = message;
        notification.className = `notification ${type} show`;
        
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }
    
    // Utility functions
    formatNumber(num) {
        return new Intl.NumberFormat('fa-IR').format(num);
    }
    
    formatCurrency(num) {
        return this.formatNumber(num) + ' ریال';
    }
    
    numberToPersianWords(num) {
        // تبدیل ساده اعداد به حروف - برای نسخه واقعی از کتابخانه persian-tools استفاده کنید
        if (num >= 1000000000) {
            const billions = Math.floor(num / 1000000000);
            const remainder = num % 1000000000;
            return `${this.formatNumber(billions)} میلیارد و ${this.numberToPersianWords(remainder)}`;
        }
        
        if (num >= 1000000) {
            const millions = Math.floor(num / 1000000);
            const remainder = num % 1000000;
            return `${this.formatNumber(millions)} میلیون و ${this.numberToPersianWords(remainder)}`;
        }
        
        if (num >= 1000) {
            const thousands = Math.floor(num / 1000);
            const remainder = num % 1000;
            return `${this.formatNumber(thousands)} هزار و ${this.numberToPersianWords(remainder)}`;
        }
        
        return this.formatNumber(num) + ' تومان';
    }
}

// Initialize the app when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.invoiceApp = new InvoiceGeneratorPro();
});

// Add notification styles dynamically
const notificationStyles = `
    .notification.success {
        background: linear-gradient(135deg, #2a9d8f, #1d7873);
    }
    .notification.error {
        background: linear-gradient(135deg, #e63946, #d62839);
    }
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);
