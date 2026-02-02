// فاکتورساز حرفه‌ای - اسکریپت اصلی
class InvoiceApp {
    constructor() {
        this.config = {
            template: 'modern',
            color: '#4361ee',
            dateFormat: 'jalali',
            fontSize: 16
        };
        
        // محدودیت مبلغ (20 میلیون)
        this.MAX_AMOUNT = 20000000;
        
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
        this.preventZoom();
        this.setupEventListeners();
        this.renderInvoice();
        this.renderItems();
        this.updateTotals();
        this.checkAmountLimits();
    }
    
    preventZoom() {
        // جلوگیری از زوم دو انگشتی
        document.addEventListener('touchmove', function(event) {
            if (event.scale !== 1) {
                event.preventDefault();
            }
        }, { passive: false });
        
        document.addEventListener('gesturestart', function(e) {
            e.preventDefault();
        });
        
        document.addEventListener('dblclick', function(e) {
            e.preventDefault();
        }, { passive: false });
    }
    
    setupEventListeners() {
        // Template selection
        document.querySelectorAll('.template-option').forEach(option => {
            option.addEventListener('click', () => {
                document.querySelectorAll('.template-option').forEach(o => o.classList.remove('active'));
                option.classList.add('active');
                this.config.template = option.dataset.template;
                this.renderInvoice();
            });
        });
        
        // Color selection
        document.querySelectorAll('.color-option').forEach(option => {
            option.addEventListener('click', () => {
                document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
                option.classList.add('active');
                this.config.color = option.dataset.color;
                document.getElementById('customColor').value = this.config.color;
                this.renderInvoice();
            });
        });
        
        // Custom color
        document.getElementById('customColor').addEventListener('input', (e) => {
            this.config.color = e.target.value;
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            this.renderInvoice();
        });
        
        // Date format
        document.querySelectorAll('input[name="dateFormat"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.config.dateFormat = e.target.value;
                this.renderInvoice();
            });
        });
        
        // Font size
        document.getElementById('fontSize').addEventListener('input', (e) => {
            this.config.fontSize = parseInt(e.target.value);
            document.getElementById('currentFontSize').textContent = this.config.fontSize;
            this.renderInvoice();
        });
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                
                // Update active tab
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Show active tab content
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(tab + 'Tab').classList.add('active');
            });
        });
        
        // Input fields
        ['sellerName', 'sellerPhone', 'sellerAddress', 'buyerName', 'buyerPhone', 'invoiceNumber']
            .forEach(id => {
                const element = document.getElementById(id);
                element.addEventListener('input', () => {
                    this.renderInvoice();
                    this.checkAmountLimits();
                });
                
                // Add validation
                if (id.includes('Price') || id.includes('Amount')) {
                    element.addEventListener('blur', () => this.validateAmount(element));
                }
            });
        
        // Add item button
        document.getElementById('addItemBtn').addEventListener('click', () => this.addNewItem());
        
        // Download button
        document.getElementById('downloadBtn').addEventListener('click', () => this.downloadInvoice());
        
        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => this.resetToDefaults());
        
        // Help button
        document.getElementById('helpBtn').addEventListener('click', () => {
            document.getElementById('helpModal').classList.add('active');
        });
        
        // Close modal
        document.querySelector('.modal-close').addEventListener('click', () => {
            document.getElementById('helpModal').classList.remove('active');
        });
        
        // Close modal on outside click
        document.getElementById('helpModal').addEventListener('click', (e) => {
            if (e.target === document.getElementById('helpModal')) {
                document.getElementById('helpModal').classList.remove('active');
            }
        });
        
        // Terms and contact buttons
        document.getElementById('termsBtn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showNotification('صفحه شرایط استفاده به زودی اضافه می‌شود', 'info');
        });
        
        document.getElementById('contactBtn')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showNotification('صفحه تماس با ما به زودی اضافه می‌شود', 'info');
        });
    }
    
    validateAmount(element) {
        const value = parseFloat(element.value) || 0;
        if (value > this.MAX_AMOUNT) {
            this.showNotification(`مبلغ نمی‌تواند بیشتر از ${this.formatNumber(this.MAX_AMOUNT)} ریال باشد`, 'error');
            element.value = this.MAX_AMOUNT;
            element.focus();
        }
    }
    
    checkAmountLimits() {
        let totalAmount = 0;
        
        this.items.forEach(item => {
            const itemTotal = item.quantity * item.unitPrice * (1 - item.discount / 100);
            totalAmount += itemTotal;
        });
        
        if (totalAmount > this.MAX_AMOUNT) {
            this.showNotification(`مجموع مبلغ فاکتور از ${this.formatNumber(this.MAX_AMOUNT)} ریال بیشتر شد!`, 'error');
            return false;
        }
        
        return true;
    }
    
    renderInvoice() {
        const data = this.getInvoiceData();
        const html = this.generateInvoiceHTML(data);
        document.getElementById('invoicePreview').innerHTML = html;
        
        // Update preview count
        const count = parseInt(document.getElementById('previewCount').textContent) || 0;
        document.getElementById('previewCount').textContent = count + 1;
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
        // تبدیل تاریخ میلادی به شمسی (ساده شده)
        const jalaliMonths = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 
                             'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'];
        const jalaliYear = 1403;
        const monthIndex = date.getMonth();
        const month = jalaliMonths[monthIndex];
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
            <div class="invoice" style="font-size: ${config.fontSize}px; color: #333;">
                <!-- Header -->
                <div class="invoice-header" style="border-bottom: 3px solid ${config.color}; padding-bottom: 20px; margin-bottom: 30px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h1 style="color: ${config.color}; font-size: 2em; margin-bottom: 10px;">فاکتور فروش</h1>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px;">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                    <span>شماره:</span>
                                    <strong>${invoiceNumber}</strong>
                                </div>
                                <div style="display: flex; justify-content: space-between;">
                                    <span>تاریخ:</span>
                                    <strong>${date}</strong>
                                </div>
                            </div>
                        </div>
                        <div style="opacity: 0.1; font-size: 3em; font-weight: 900; transform: rotate(-15deg);">
                            ${seller.name}
                        </div>
                    </div>
                </div>
                
                <!-- Parties -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
                    <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-right: 5px solid ${config.color};">
                        <h3 style="color: ${config.color}; margin-bottom: 15px; font-size: 1.2em;">فروشنده</h3>
                        <div>
                            <p style="margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                                <i class="fas fa-store"></i> ${seller.name}
                            </p>
                            <p style="margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                                <i class="fas fa-phone"></i> ${seller.phone}
                            </p>
                            <p style="display: flex; align-items: center; gap: 10px;">
                                <i class="fas fa-map-marker-alt"></i> ${seller.address}
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-right: 5px solid ${config.color};">
                        <h3 style="color: ${config.color}; margin-bottom: 15px; font-size: 1.2em;">خریدار</h3>
                        <div>
                            <p style="margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                                <i class="fas fa-user"></i> ${buyer.name}
                            </p>
                            <p style="display: flex; align-items: center; gap: 10px;">
                                <i class="fas fa-mobile-alt"></i> ${buyer.phone}
                            </p>
                        </div>
                    </div>
                </div>
                
                <!-- Items Table -->
                <table style="width: 100%; border-collapse: collapse; margin: 30px 0; font-family: 'Vazirmatn', sans-serif;">
                    <thead>
                        <tr style="background: ${config.color}; color: white;">
                            <th style="padding: 15px; text-align: right;">ردیف</th>
                            <th style="padding: 15px; text-align: right;">شرح کالا / خدمت</th>
                            <th style="padding: 15px; text-align: right;">تعداد</th>
                            <th style="padding: 15px; text-align: right;">قیمت واحد (ریال)</th>
                            <th style="padding: 15px; text-align: right;">تخفیف %</th>
                            <th style="padding: 15px; text-align: right;">مجموع (ریال)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${items.map((item, index) => {
                            const total = item.quantity * item.unitPrice;
                            const afterDiscount = total * (1 - item.discount/100);
                            const rowStyle = index % 2 === 0 ? 'background: #fafafa;' : '';
                            
                            return `
                                <tr style="${rowStyle} border-bottom: 1px solid #eee;">
                                    <td style="padding: 12px 15px;">${index + 1}</td>
                                    <td style="padding: 12px 15px;">${item.description}</td>
                                    <td style="padding: 12px 15px;">${item.quantity}</td>
                                    <td style="padding: 12px 15px;">${this.formatNumber(item.unitPrice)}</td>
                                    <td style="padding: 12px 15px;">${item.discount}%</td>
                                    <td style="padding: 12px 15px;">${this.formatNumber(afterDiscount)}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
                
                <!-- Summary -->
                <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; margin: 40px 0; border: 2px dashed ${config.color}40;">
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>جمع کل:</span>
                        <span>${this.formatCurrency(totals.subtotal)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>تخفیف:</span>
                        <span>${this.formatCurrency(totals.discount)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>مالیات بر ارزش افزوده (۹٪):</span>
                        <span>${this.formatCurrency(totals.tax)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 15px 0; font-size: 1.3em; font-weight: 900; border-top: 2px solid ${config.color}; margin-top: 10px;">
                        <span>مبلغ قابل پرداخت:</span>
                        <span style="color: ${config.color};">${this.formatCurrency(totals.total)}</span>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin-top: 20px; border-right: 4px solid ${config.color};">
                        <strong>مبلغ به حروف:</strong> ${totals.totalInWords}
                    </div>
                </div>
                
                <!-- Footer -->
                <div style="margin-top: 50px; padding-top: 30px; border-top: 2px solid #eee;">
                    <div style="margin-bottom: 30px;">
                        <p><strong>توضیحات:</strong> کالاها ظرف ۳ روز کاری آماده تحویل می‌باشد. گارانتی ۱۸ ماهه</p>
                        <p><strong>روش پرداخت:</strong> نقدی</p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 50px; margin: 40px 0;">
                        <div style="text-align: center;">
                            <p>امضای خریدار</p>
                            <div style="height: 2px; background: #333; margin-top: 60px; position: relative;">
                                <div style="position: absolute; width: 100%; height: 20px; border-bottom: 1px dashed #666; bottom: -30px;"></div>
                            </div>
                        </div>
                        <div style="text-align: center;">
                            <p>امضا و مهر فروشنده</p>
                            <div style="height: 2px; background: #333; margin-top: 60px; position: relative;">
                                <div style="position: absolute; width: 100%; height: 20px; border-bottom: 1px dashed #666; bottom: -30px;"></div>
                            </div>
                        </div>
                    </div>
                    
                    <div style="text-align: center; color: #666; font-style: italic; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                        این فاکتور به صورت الکترونیکی صادر شده و نیاز به مهر و امضای فیزیکی ندارد
                    </div>
                </div>
            </div>
        `;
    }
    
    renderItems() {
        const container = document.getElementById('itemsList');
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
        // Item input events
        document.querySelectorAll('.item-desc, .item-qty, .item-price, .item-discount').forEach(input => {
            input.addEventListener('input', (e) => {
                const index = parseInt(e.target.dataset.index);
                const field = e.target.dataset.field;
                let value = e.target.type === 'number' ? parseFloat(e.target.value) || 0 : e.target.value;
                
                // Validate amount limit
                if ((field === 'unitPrice' || field === 'quantity') && value > this.MAX_AMOUNT) {
                    value = this.MAX_AMOUNT;
                    e.target.value = value;
                    this.showNotification(`مبلغ نمی‌تواند بیشتر از ${this.formatNumber(this.MAX_AMOUNT)} ریال باشد`, 'error');
                }
                
                this.items[index][field] = value;
                this.renderInvoice();
                this.updateTotals();
                this.checkAmountLimits();
                
                // Update total for this row
                if (field !== 'description') {
                    const item = this.items[index];
                    const total = item.quantity * item.unitPrice * (1 - item.discount/100);
                    e.target.closest('.item-row').querySelector('.item-total').textContent = this.formatCurrency(total);
                }
            });
        });
        
        // Remove item buttons
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
    }
    
    async downloadInvoice() {
        // Check amount limit before download
        if (!this.checkAmountLimits()) {
            return;
        }
        
        const btn = document.getElementById('downloadBtn');
        const originalHTML = btn.innerHTML;
        
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال ایجاد...';
        btn.disabled = true;
        
        try {
            // Wait for fonts to load
            await document.fonts.ready;
            
            const element = document.querySelector('#invoicePreview .invoice');
            const canvas = await html2canvas(element, {
                scale: 2,
                useCORS: true,
                backgroundColor: '#ffffff',
                logging: false
            });
            
            const link = document.createElement('a');
            link.download = `فاکتور-${document.getElementById('invoiceNumber').value || 'جدید'}.png`;
            link.href = canvas.toDataURL('image/png');
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
            
            // Reset UI elements
            document.querySelectorAll('.template-option').forEach(o => o.classList.remove('active'));
            document.querySelector('.template-option[data-template="modern"]').classList.add('active');
            
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            document.querySelector('.color-option[data-color="#4361ee"]').classList.add('active');
            
            document.getElementById('customColor').value = '#4361ee';
            
            document.querySelector('input[name="dateFormat"][value="jalali"]').checked = true;
            
            document.getElementById('fontSize').value = 16;
            document.getElementById('currentFontSize').textContent = '16';
            
            // Reset input fields
            document.getElementById('sellerName').value = 'فروشگاه موبایل پارسیان';
            document.getElementById('sellerPhone').value = '021-12345678';
            document.getElementById('sellerAddress').value = 'تهران، خیابان ولیعصر، پلاک ۲۴۰';
            document.getElementById('buyerName').value = 'سعید کنانی';
            document.getElementById('buyerPhone').value = '09137657870';
            document.getElementById('invoiceNumber').value = 'INV-1403-001';
            
            // Reset tabs
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.tab-btn[data-tab="seller"]').classList.add('active');
            
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById('sellerTab').classList.add('active');
            
            this.renderInvoice();
            this.renderItems();
            this.updateTotals();
            
            this.showNotification('همه تنظیمات بازنشانی شد', 'success');
        }
    }
    
    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        const text = document.getElementById('notificationText');
        
        if (!notification) {
            // Create notification element if it doesn't exist
            const notifEl = document.createElement('div');
            notifEl.id = 'notification';
            notifEl.className = `notification ${type}`;
            notifEl.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                <span id="notificationText">${message}</span>
            `;
            document.body.appendChild(notifEl);
            
            setTimeout(() => {
                notifEl.classList.add('show');
                setTimeout(() => {
                    notifEl.classList.remove('show');
                    setTimeout(() => notifEl.remove(), 300);
                }, 3000);
            }, 10);
            return;
        }
        
        // Update existing notification
        const icon = notification.querySelector('i');
        icon.className = `fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}`;
        text.textContent = message;
        notification.className = `notification ${type}`;
        
        // Show notification
        setTimeout(() => {
            notification.classList.add('show');
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }, 10);
    }
    
    // Utility functions
    formatNumber(num) {
        return new Intl.NumberFormat('fa-IR').format(num);
    }
    
    formatCurrency(num) {
        return this.formatNumber(num) + ' ریال';
    }
    
    numberToPersianWords(num) {
        // تبدیل اعداد به حروف فارسی (ساده شده)
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
    window.invoiceApp = new InvoiceApp();
});
