// فاکتورساز حرفه‌ای - جاوااسکریپت اصلاح شده
document.addEventListener('DOMContentLoaded', function() {
    // جلوگیری از زوم دو انگشتی
    document.addEventListener('touchmove', function(event) {
        if (event.scale !== 1) {
            event.preventDefault();
        }
    }, { passive: false });
    
    document.addEventListener('gesturestart', function(e) {
        e.preventDefault();
    });
    
    // تنظیمات
    let config = {
        template: 'modern',
        color: '#4361ee',
        dateFormat: 'jalali',
        fontSize: 16
    };
    
    const MAX_AMOUNT = 20000000; // 20 میلیون
    
    let items = [
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
    
    // عناصر DOM
    const elements = {
        invoicePreview: document.getElementById('invoicePreview'),
        itemsList: document.getElementById('itemsList'),
        subtotal: document.getElementById('subtotal'),
        discountAmount: document.getElementById('discountAmount'),
        taxAmount: document.getElementById('taxAmount'),
        totalAmount: document.getElementById('totalAmount'),
        previewCount: document.getElementById('previewCount'),
        customColor: document.getElementById('customColor'),
        fontSize: document.getElementById('fontSize'),
        currentFontSize: document.getElementById('currentFontSize'),
        sellerName: document.getElementById('sellerName'),
        sellerPhone: document.getElementById('sellerPhone'),
        sellerAddress: document.getElementById('sellerAddress'),
        buyerName: document.getElementById('buyerName'),
        buyerPhone: document.getElementById('buyerPhone'),
        invoiceNumber: document.getElementById('invoiceNumber')
    };
    
    // شمارنده پیش‌نمایش
    let previewCounter = 1;
    
    // راه‌اندازی اولیه
    init();
    
    function init() {
        setupEventListeners();
        renderInvoice();
        renderItems();
        updateTotals();
        makePanelsSticky();
    }
    
    function makePanelsSticky() {
        // پنل کنترل sticky
        const controlPanel = document.querySelector('.control-panel');
        const dataPanel = document.querySelector('.data-panel');
        
        if (controlPanel) {
            window.addEventListener('scroll', function() {
                const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                
                if (scrollTop > 100) {
                    controlPanel.style.position = 'sticky';
                    controlPanel.style.top = '20px';
                    controlPanel.style.zIndex = '10';
                    
                    if (dataPanel) {
                        dataPanel.style.position = 'sticky';
                        dataPanel.style.top = '20px';
                        dataPanel.style.zIndex = '10';
                    }
                } else {
                    controlPanel.style.position = 'relative';
                    controlPanel.style.top = 'auto';
                    
                    if (dataPanel) {
                        dataPanel.style.position = 'relative';
                        dataPanel.style.top = 'auto';
                    }
                }
            });
        }
    }
    
    function setupEventListeners() {
        // انتخاب قالب
        document.querySelectorAll('.template-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.template-option').forEach(o => o.classList.remove('active'));
                this.classList.add('active');
                config.template = this.dataset.template;
                renderInvoice();
            });
        });
        
        // انتخاب رنگ
        document.querySelectorAll('.color-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
                this.classList.add('active');
                config.color = this.dataset.color;
                elements.customColor.value = config.color;
                renderInvoice();
            });
        });
        
        // رنگ دلخواه
        elements.customColor.addEventListener('input', function(e) {
            config.color = e.target.value;
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            renderInvoice();
        });
        
        // فرمت تاریخ
        document.querySelectorAll('input[name="dateFormat"]').forEach(radio => {
            radio.addEventListener('change', function(e) {
                config.dateFormat = e.target.value;
                renderInvoice();
            });
        });
        
        // اندازه فونت
        elements.fontSize.addEventListener('input', function(e) {
            config.fontSize = parseInt(e.target.value);
            elements.currentFontSize.textContent = config.fontSize;
            renderInvoice();
        });
        
        // تب‌ها
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const tab = this.dataset.tab;
                
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(tab + 'Tab').classList.add('active');
            });
        });
        
        // فیلدهای ورودی
        ['sellerName', 'sellerPhone', 'sellerAddress', 'buyerName', 'buyerPhone', 'invoiceNumber']
            .forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    element.addEventListener('input', renderInvoice);
                }
            });
        
        // افزودن آیتم
        document.getElementById('addItemBtn').addEventListener('click', addNewItem);
        
        // دانلود
        document.getElementById('downloadBtn').addEventListener('click', downloadInvoice);
        
        // بازنشانی
        document.getElementById('resetBtn').addEventListener('click', resetToDefaults);
        
        // راهنما
        const helpBtn = document.getElementById('helpBtn');
        const helpModal = document.getElementById('helpModal');
        const modalClose = document.querySelector('.modal-close');
        
        if (helpBtn && helpModal && modalClose) {
            helpBtn.addEventListener('click', function() {
                helpModal.classList.add('active');
            });
            
            modalClose.addEventListener('click', function() {
                helpModal.classList.remove('active');
            });
            
            helpModal.addEventListener('click', function(e) {
                if (e.target === helpModal) {
                    helpModal.classList.remove('active');
                }
            });
        }
    }
    
    function renderInvoice() {
        const data = getInvoiceData();
        elements.invoicePreview.innerHTML = generateInvoiceHTML(data);
        
        // افزایش شمارنده
        previewCounter++;
        if (elements.previewCount) {
            elements.previewCount.textContent = previewCounter;
        }
    }
    
    function getInvoiceData() {
        return {
            seller: {
                name: elements.sellerName.value || 'فروشگاه موبایل پارسیان',
                phone: elements.sellerPhone.value || '021-12345678',
                address: elements.sellerAddress.value || 'تهران، خیابان ولیعصر، پلاک ۲۴۰'
            },
            buyer: {
                name: elements.buyerName.value || 'سعید کنانی',
                phone: elements.buyerPhone.value || '09137657870'
            },
            invoiceNumber: elements.invoiceNumber.value || 'INV-1403-001',
            date: getFormattedDate(),
            items: items,
            totals: calculateTotals(),
            config: config
        };
    }
    
    function getFormattedDate() {
        const now = new Date();
        const jalali = toJalali(now);
        const gregorian = toGregorian(now);
        
        switch(config.dateFormat) {
            case 'jalali': return jalali;
            case 'gregorian': return gregorian;
            case 'both': return `${jalali} | ${gregorian}`;
            default: return jalali;
        }
    }
    
    function toJalali(date) {
        const jalaliMonths = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 
                             'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'];
        const jalaliYear = 1403;
        const monthIndex = date.getMonth();
        const month = jalaliMonths[monthIndex];
        const day = date.getDate();
        return `${jalaliYear}/${month}/${day}`;
    }
    
    function toGregorian(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}/${month}/${day}`;
    }
    
    function calculateTotals() {
        let subtotal = 0;
        let totalDiscount = 0;
        
        items.forEach(item => {
            const itemTotal = item.quantity * item.unitPrice;
            subtotal += itemTotal;
            totalDiscount += itemTotal * (item.discount / 100);
        });
        
        // محدودیت 20 میلیون
        if (subtotal > MAX_AMOUNT) {
            subtotal = MAX_AMOUNT;
            showNotification(`مجموع مبلغ از ${formatNumber(MAX_AMOUNT)} ریال بیشتر شد!`, 'error');
        }
        
        const tax = (subtotal - totalDiscount) * 0.09;
        const total = subtotal - totalDiscount + tax;
        
        return {
            subtotal,
            discount: totalDiscount,
            tax,
            total,
            totalInWords: numberToPersianWords(total)
        };
    }
    
    function generateInvoiceHTML(data) {
        const { seller, buyer, invoiceNumber, date, items, totals, config } = data;
        
        // تولید HTML جدول آیتم‌ها
        let itemsHTML = '';
        items.forEach((item, index) => {
            const total = item.quantity * item.unitPrice;
            const afterDiscount = total * (1 - item.discount/100);
            const rowStyle = index % 2 === 0 ? 'background: #fafafa;' : '';
            
            itemsHTML += `
                <tr style="${rowStyle} border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 15px;">${index + 1}</td>
                    <td style="padding: 12px 15px;">${item.description}</td>
                    <td style="padding: 12px 15px;">${item.quantity}</td>
                    <td style="padding: 12px 15px;">${formatNumber(item.unitPrice)}</td>
                    <td style="padding: 12px 15px;">${item.discount}%</td>
                    <td style="padding: 12px 15px;">${formatNumber(afterDiscount)}</td>
                </tr>
            `;
        });
        
        return `
            <div class="invoice" style="font-size: ${config.fontSize}px; color: #333; font-family: 'Vazirmatn', sans-serif;">
                <!-- Header -->
                <div style="border-bottom: 3px solid ${config.color}; padding-bottom: 20px; margin-bottom: 30px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div>
                            <h1 style="color: ${config.color}; font-size: 2em; margin-bottom: 10px; font-weight: 900;">فاکتور فروش</h1>
                            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; max-width: 300px;">
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
                        <div style="opacity: 0.1; font-size: 3em; font-weight: 900; transform: rotate(-15deg); user-select: none;">
                            ${seller.name}
                        </div>
                    </div>
                </div>
                
                <!-- Parties -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin-bottom: 40px;">
                    <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-right: 5px solid ${config.color};">
                        <h3 style="color: ${config.color}; margin-bottom: 15px; font-size: 1.2em; font-weight: 700;">فروشنده</h3>
                        <div>
                            <p style="margin-bottom: 10px;">
                                <i class="fas fa-store" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${seller.name}
                            </p>
                            <p style="margin-bottom: 10px;">
                                <i class="fas fa-phone" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${seller.phone}
                            </p>
                            <p>
                                <i class="fas fa-map-marker-alt" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${seller.address}
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; border-right: 5px solid ${config.color};">
                        <h3 style="color: ${config.color}; margin-bottom: 15px; font-size: 1.2em; font-weight: 700;">خریدار</h3>
                        <div>
                            <p style="margin-bottom: 10px;">
                                <i class="fas fa-user" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${buyer.name}
                            </p>
                            <p>
                                <i class="fas fa-mobile-alt" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${buyer.phone}
                            </p>
                        </div>
                    </div>
                </div>
                
                <!-- Items Table -->
                <table style="width: 100%; border-collapse: collapse; margin: 30px 0;">
                    <thead>
                        <tr style="background: ${config.color}; color: white;">
                            <th style="padding: 15px; text-align: right; font-weight: 700;">ردیف</th>
                            <th style="padding: 15px; text-align: right; font-weight: 700;">شرح کالا / خدمت</th>
                            <th style="padding: 15px; text-align: right; font-weight: 700;">تعداد</th>
                            <th style="padding: 15px; text-align: right; font-weight: 700;">قیمت واحد (ریال)</th>
                            <th style="padding: 15px; text-align: right; font-weight: 700;">تخفیف %</th>
                            <th style="padding: 15px; text-align: right; font-weight: 700;">مجموع (ریال)</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${itemsHTML}
                    </tbody>
                </table>
                
                <!-- Summary -->
                <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; margin: 40px 0; border: 2px dashed ${config.color}40;">
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>جمع کل:</span>
                        <span>${formatCurrency(totals.subtotal)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>تخفیف:</span>
                        <span>${formatCurrency(totals.discount)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>مالیات بر ارزش افزوده (۹٪):</span>
                        <span>${formatCurrency(totals.tax)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 15px 0; font-size: 1.3em; font-weight: 900; border-top: 2px solid ${config.color}; margin-top: 10px;">
                        <span>مبلغ قابل پرداخت:</span>
                        <span style="color: ${config.color};">${formatCurrency(totals.total)}</span>
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
    
    function renderItems() {
        if (!elements.itemsList) return;
        
        elements.itemsList.innerHTML = '';
        
        items.forEach((item, index) => {
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
                <span class="item-total">${formatCurrency(item.quantity * item.unitPrice * (1 - item.discount/100))}</span>
                <button class="remove-item" data-index="${index}" ${items.length <= 1 ? 'disabled' : ''}>
                    <i class="fas fa-trash"></i>
                </button>
            `;
            
            elements.itemsList.appendChild(itemElement);
        });
        
        setupItemEvents();
    }
    
    function setupItemEvents() {
        // رویدادهای آیتم‌ها
        document.querySelectorAll('.item-desc, .item-qty, .item-price, .item-discount').forEach(input => {
            input.addEventListener('input', function(e) {
                const index = parseInt(this.dataset.index);
                const field = this.dataset.field;
                let value = this.type === 'number' ? parseFloat(this.value) || 0 : this.value;
                
                // اعتبارسنجی مبلغ
                if ((field === 'unitPrice' || field === 'quantity') && value > MAX_AMOUNT) {
                    value = MAX_AMOUNT;
                    this.value = value;
                    showNotification(`مبلغ نمی‌تواند بیشتر از ${formatNumber(MAX_AMOUNT)} ریال باشد`, 'error');
                }
                
                items[index][field] = value;
                renderInvoice();
                updateTotals();
                
                // به‌روزرسانی جمع سطر
                if (field !== 'description') {
                    const item = items[index];
                    const total = item.quantity * item.unitPrice * (1 - item.discount/100);
                    this.closest('.item-row').querySelector('.item-total').textContent = formatCurrency(total);
                }
            });
        });
        
        // دکمه حذف
        document.querySelectorAll('.remove-item').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = parseInt(this.dataset.index);
                if (items.length > 1) {
                    items.splice(index, 1);
                    renderItems();
                    renderInvoice();
                    updateTotals();
                    showNotification('آیتم حذف شد', 'success');
                }
            });
        });
    }
    
    function addNewItem() {
        const newItem = {
            id: items.length + 1,
            description: 'کالا یا خدمات جدید',
            quantity: 1,
            unitPrice: 1000000,
            discount: 0
        };
        
        items.push(newItem);
        renderItems();
        renderInvoice();
        updateTotals();
        showNotification('آیتم جدید اضافه شد', 'success');
    }
    
    function updateTotals() {
        const totals = calculateTotals();
        
        if (elements.subtotal) elements.subtotal.textContent = formatCurrency(totals.subtotal);
        if (elements.discountAmount) elements.discountAmount.textContent = formatCurrency(totals.discount);
        if (elements.taxAmount) elements.taxAmount.textContent = formatCurrency(totals.tax);
        if (elements.totalAmount) elements.totalAmount.textContent = formatCurrency(totals.total);
    }
    
    function downloadInvoice() {
        const btn = document.getElementById('downloadBtn');
        const originalHTML = btn.innerHTML;
        
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال ایجاد...';
        btn.disabled = true;
        
        // بررسی وجود html2canvas
        if (typeof html2canvas === 'undefined') {
            showNotification('خطا: کتابخانه html2canvas بارگذاری نشده است', 'error');
            btn.innerHTML = originalHTML;
            btn.disabled = false;
            return;
        }
        
        // انتخاب المنت دقیق
        const invoiceElement = elements.invoicePreview.querySelector('.invoice') || elements.invoicePreview;
        
        // گزینه‌های html2canvas
        const options = {
            scale: 2,
            useCORS: true,
            allowTaint: true,
            backgroundColor: '#ffffff',
            logging: false,
            onclone: function(clonedDoc) {
                // بارگذاری فونت‌ها در نسخه کلون شده
                const style = clonedDoc.createElement('style');
                style.textContent = `
                    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap');
                    * {
                        font-family: 'Vazirmatn', sans-serif !important;
                    }
                `;
                clonedDoc.head.appendChild(style);
            }
        };
        
        // ایجاد تصویر
        setTimeout(() => {
            html2canvas(invoiceElement, options)
                .then(canvas => {
                    // ایجاد لینک دانلود
                    const link = document.createElement('a');
                    link.download = `فاکتور-${elements.invoiceNumber.value || 'جدید'}.png`;
                    link.href = canvas.toDataURL('image/png');
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    showNotification('فاکتور با موفقیت دانلود شد!', 'success');
                })
                .catch(error => {
                    console.error('خطا در ایجاد تصویر:', error);
                    showNotification('خطا در ایجاد تصویر. لطفاً دوباره امتحان کنید', 'error');
                })
                .finally(() => {
                    btn.innerHTML = originalHTML;
                    btn.disabled = false;
                });
        }, 500);
    }
    
    function resetToDefaults() {
        if (confirm('آیا از بازنشانی همه تنظیمات به حالت اولیه اطمینان دارید؟')) {
            config = {
                template: 'modern',
                color: '#4361ee',
                dateFormat: 'jalali',
                fontSize: 16
            };
            
            items = [
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
            
            // بازنشانی UI
            document.querySelectorAll('.template-option').forEach(o => o.classList.remove('active'));
            document.querySelector('.template-option[data-template="modern"]').classList.add('active');
            
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            document.querySelector('.color-option[data-color="#4361ee"]').classList.add('active');
            
            elements.customColor.value = '#4361ee';
            
            document.querySelector('input[name="dateFormat"][value="jalali"]').checked = true;
            
            elements.fontSize.value = 16;
            elements.currentFontSize.textContent = '16';
            
            // بازنشانی فیلدها
            elements.sellerName.value = 'فروشگاه موبایل پارسیان';
            elements.sellerPhone.value = '021-12345678';
            elements.sellerAddress.value = 'تهران، خیابان ولیعصر، پلاک ۲۴۰';
            elements.buyerName.value = 'سعید کنانی';
            elements.buyerPhone.value = '09137657870';
            elements.invoiceNumber.value = 'INV-1403-001';
            
            // بازنشانی تب‌ها
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.tab-btn[data-tab="seller"]').classList.add('active');
            
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById('sellerTab').classList.add('active');
            
            // رندر مجدد
            renderInvoice();
            renderItems();
            updateTotals();
            
            showNotification('همه تنظیمات بازنشانی شد', 'success');
        }
    }
    
    // تابع‌های کمکی
    function formatNumber(num) {
        return new Intl.NumberFormat('fa-IR').format(num);
    }
    
    function formatCurrency(num) {
        return formatNumber(num) + ' ریال';
    }
    
    function numberToPersianWords(num) {
        // تبدیل ساده اعداد به حروف
        if (num >= 1000000000) {
            const billions = Math.floor(num / 1000000000);
            const remainder = num % 1000000000;
            return `${formatNumber(billions)} میلیارد و ${numberToPersianWords(remainder)}`;
        }
        
        if (num >= 1000000) {
            const millions = Math.floor(num / 1000000);
            const remainder = num % 1000000;
            return `${formatNumber(millions)} میلیون و ${numberToPersianWords(remainder)}`;
        }
        
        if (num >= 1000) {
            const thousands = Math.floor(num / 1000);
            const remainder = num % 1000;
            return `${formatNumber(thousands)} هزار و ${numberToPersianWords(remainder)}`;
        }
        
        return formatNumber(num) + ' تومان';
    }
    
    function showNotification(message, type = 'success') {
        // ایجاد یا به‌روزرسانی نوتیفیکیشن
        let notification = document.getElementById('notification');
        
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'notification';
            notification.className = `notification ${type}`;
            notification.innerHTML = `
                <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                <span id="notificationText">${message}</span>
            `;
            document.body.appendChild(notification);
        } else {
            const icon = notification.querySelector('i');
            icon.className = `fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}`;
            notification.querySelector('#notificationText').textContent = message;
            notification.className = `notification ${type}`;
        }
        
        // نمایش نوتیفیکیشن
        setTimeout(() => {
            notification.classList.add('show');
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }, 10);
    }
    
    // بارگذاری html2canvas اگر موجود نیست
    if (typeof html2canvas === 'undefined') {
        const script = document.createElement('script');
        script.src = 'https://html2canvas.hertzen.com/dist/html2canvas.min.js';
        script.onload = function() {
            console.log('html2canvas loaded successfully');
        };
        script.onerror = function() {
            showNotification('خطا در بارگذاری کتابخانه ایجاد تصویر', 'error');
        };
        document.head.appendChild(script);
    }
});
