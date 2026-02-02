// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Set initial values
    let currentTemplate = 'modern';
    let currentColor = '#4361ee';
    let currentDateFormat = 'jalali';
    let fontSize = 14;
    
    // Initialize items
    const initialItems = [
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
    
    let items = [...initialItems];
    
    // DOM Elements
    const invoiceContainer = document.getElementById('invoiceContainer');
    const colorPicker = document.getElementById('colorPicker');
    const templateButtons = document.querySelectorAll('.template-btn');
    const colorPresets = document.querySelectorAll('.color-preset');
    const dateFormatRadios = document.querySelectorAll('input[name="dateFormat"]');
    const fontSizeSlider = document.getElementById('fontSize');
    const itemsList = document.getElementById('itemsList');
    const addItemBtn = document.getElementById('addItemBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const resetBtn = document.getElementById('resetBtn');
    
    // Initialize UI
    updateInvoice();
    renderItems();
    calculateTotals();
    
    // Event Listeners
    colorPicker.addEventListener('input', function(e) {
        currentColor = e.target.value;
        updateInvoice();
    });
    
    templateButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            // Remove active class from all buttons
            templateButtons.forEach(b => b.classList.remove('active'));
            // Add active class to clicked button
            this.classList.add('active');
            currentTemplate = this.dataset.template;
            currentColor = this.dataset.color;
            colorPicker.value = currentColor;
            updateInvoice();
        });
    });
    
    colorPresets.forEach(preset => {
        preset.addEventListener('click', function() {
            currentColor = this.dataset.color;
            colorPicker.value = currentColor;
            updateInvoice();
        });
    });
    
    dateFormatRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                currentDateFormat = this.value;
                updateInvoice();
            }
        });
    });
    
    fontSizeSlider.addEventListener('input', function(e) {
        fontSize = parseInt(e.target.value);
        updateInvoice();
    });
    
    // Input fields
    const inputFields = ['sellerName', 'sellerPhone', 'sellerAddress', 'buyerName', 'buyerPhone', 'invoiceNumber'];
    inputFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        field.addEventListener('input', updateInvoice);
    });
    
    // Add new item
    addItemBtn.addEventListener('click', function() {
        const newItem = {
            id: items.length + 1,
            description: 'کالا/خدمت جدید',
            quantity: 1,
            unitPrice: 1000000,
            discount: 0
        };
        items.push(newItem);
        renderItems();
        updateInvoice();
        calculateTotals();
    });
    
    // Download invoice as PNG
    downloadBtn.addEventListener('click', function() {
        downloadInvoice();
    });
    
    // Reset to defaults
    resetBtn.addEventListener('click', function() {
        if (confirm('آیا از بازنشانی تمام تنظیمات به حالت اولیه اطمینان دارید؟')) {
            currentTemplate = 'modern';
            currentColor = '#4361ee';
            currentDateFormat = 'jalali';
            fontSize = 14;
            
            // Reset template buttons
            templateButtons.forEach(btn => {
                btn.classList.remove('active');
                if (btn.dataset.template === 'modern') {
                    btn.classList.add('active');
                }
            });
            
            // Reset color picker
            colorPicker.value = currentColor;
            
            // Reset date format
            document.querySelector('input[name="dateFormat"][value="jalali"]').checked = true;
            
            // Reset font size
            fontSizeSlider.value = fontSize;
            
            // Reset items to initial
            items = [...initialItems];
            
            // Reset input fields
            document.getElementById('sellerName').value = 'فروشگاه موبایل پارسیان';
            document.getElementById('sellerPhone').value = '021-12345678';
            document.getElementById('sellerAddress').value = 'تهران، خیابان ولیعصر، پلاک ۲۴۰';
            document.getElementById('buyerName').value = 'سعید کنانی';
            document.getElementById('buyerPhone').value = '09137657870';
            document.getElementById('invoiceNumber').value = 'INV-1403-001';
            
            updateInvoice();
            renderItems();
            calculateTotals();
        }
    });
    
    // Render items list
    function renderItems() {
        itemsList.innerHTML = '';
        
        items.forEach((item, index) => {
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
                <span class="item-total">${formatCurrency(item.quantity * item.unitPrice * (1 - item.discount/100))}</span>
                <button class="remove-item" data-index="${index}">×</button>
            `;
            
            itemsList.appendChild(itemElement);
        });
        
        // Add event listeners to item inputs
        document.querySelectorAll('.item-desc, .item-qty, .item-price, .item-discount').forEach(input => {
            input.addEventListener('input', function() {
                const index = parseInt(this.dataset.index);
                const field = this.dataset.field;
                const value = this.type === 'number' ? parseFloat(this.value) || 0 : this.value;
                
                items[index][field] = value;
                updateInvoice();
                calculateTotals();
                
                // Update total for this row
                if (field !== 'description') {
                    const item = items[index];
                    const total = item.quantity * item.unitPrice * (1 - item.discount/100);
                    this.closest('.item-row').querySelector('.item-total').textContent = formatCurrency(total);
                }
            });
        });
        
        // Add event listeners to remove buttons
        document.querySelectorAll('.remove-item').forEach(btn => {
            btn.addEventListener('click', function() {
                const index = parseInt(this.dataset.index);
                if (items.length > 1) {
                    items.splice(index, 1);
                    renderItems();
                    updateInvoice();
                    calculateTotals();
                } else {
                    alert('حداقل یک آیتم باید وجود داشته باشد');
                }
            });
        });
    }
    
    // Calculate totals
    function calculateTotals() {
        let subtotal = 0;
        let totalDiscount = 0;
        
        items.forEach(item => {
            const itemTotal = item.quantity * item.unitPrice;
            subtotal += itemTotal;
            totalDiscount += itemTotal * (item.discount / 100);
        });
        
        const tax = (subtotal - totalDiscount) * 0.09;
        const total = subtotal - totalDiscount + tax;
        
        // Update UI
        document.getElementById('subtotal').textContent = formatCurrency(subtotal);
        document.getElementById('discountAmount').textContent = formatCurrency(totalDiscount);
        document.getElementById('taxAmount').textContent = formatCurrency(tax);
        document.getElementById('totalAmount').textContent = formatCurrency(total);
    }
    
    // Update invoice preview
    function updateInvoice() {
        const sellerName = document.getElementById('sellerName').value;
        const sellerPhone = document.getElementById('sellerPhone').value;
        const sellerAddress = document.getElementById('sellerAddress').value;
        const buyerName = document.getElementById('buyerName').value;
        const buyerPhone = document.getElementById('buyerPhone').value;
        const invoiceNumber = document.getElementById('invoiceNumber').value;
        
        // Get current date
        const now = new Date();
        const jalaliDate = getJalaliDate(now);
        const gregorianDate = getGregorianDate(now);
        
        let dateDisplay = '';
        switch(currentDateFormat) {
            case 'jalali':
                dateDisplay = jalaliDate;
                break;
            case 'gregorian':
                dateDisplay = gregorianDate;
                break;
            case 'both':
                dateDisplay = `${jalaliDate} | ${gregorianDate}`;
                break;
        }
        
        // Generate invoice HTML
        invoiceContainer.innerHTML = `
            <div class="invoice ${currentTemplate}" id="invoicePreview" style="font-size: ${fontSize}px; --accent: ${currentColor}">
                <div class="invoice-header">
                    <div class="invoice-title">فاکتور فروش</div>
                    <div class="invoice-meta">
                        <div class="invoice-number">شماره: ${invoiceNumber}</div>
                        <div class="invoice-date">تاریخ: ${dateDisplay}</div>
                    </div>
                </div>
                
                <div class="parties">
                    <div class="party seller">
                        <h3>فروشنده:</h3>
                        <div class="party-info">
                            <p><i class="fas fa-store"></i> ${sellerName}</p>
                            <p><i class="fas fa-phone"></i> ${sellerPhone}</p>
                            <p><i class="fas fa-map-marker-alt"></i> ${sellerAddress}</p>
                        </div>
                    </div>
                    
                    <div class="party buyer">
                        <h3>خریدار:</h3>
                        <div class="party-info">
                            <p><i class="fas fa-user"></i> ${buyerName}</p>
                            <p><i class="fas fa-mobile-alt"></i> ${buyerPhone}</p>
                        </div>
                    </div>
                </div>
                
                <table class="items-table">
                    <thead>
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
                                <tr>
                                    <td>${index + 1}</td>
                                    <td>${item.description}</td>
                                    <td>${item.quantity}</td>
                                    <td>${formatNumber(item.unitPrice)}</td>
                                    <td>${item.discount}%</td>
                                    <td>${formatNumber(afterDiscount)}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
                
                <div class="summary">
                    ${generateSummary()}
                </div>
                
                <div class="footer-notes">
                    <p><strong>توضیحات:</strong> کالاها ظرف ۳ روز کاری آماده تحویل می‌باشد. گارانتی ۱۸ ماهه</p>
                    <p><strong>روش پرداخت:</strong> نقدی</p>
                    <div class="signatures">
                        <div class="signature">
                            <p>امضای خریدار</p>
                        </div>
                        <div class="signature">
                            <p>امضا و مهر فروشنده</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add styles based on template
        const style = document.createElement('style');
        style.textContent = `
            .invoice.modern {
                --accent: ${currentColor};
            }
            .invoice.classic {
                --accent: ${currentColor};
                font-family: 'Vazirmatn', serif;
            }
            .invoice.minimal {
                --accent: ${currentColor};
                border: 1px solid ${currentColor};
            }
        `;
        invoiceContainer.appendChild(style);
    }
    
    // Generate summary section
    function generateSummary() {
        let subtotal = 0;
        let totalDiscount = 0;
        
        items.forEach(item => {
            const itemTotal = item.quantity * item.unitPrice;
            subtotal += itemTotal;
            totalDiscount += itemTotal * (item.discount / 100);
        });
        
        const tax = (subtotal - totalDiscount) * 0.09;
        const total = subtotal - totalDiscount + tax;
        
        // Convert to words (simplified)
        const totalInWords = convertToPersianWords(total);
        
        return `
            <div class="summary-row">
                <span>جمع کل:</span>
                <span>${formatCurrency(subtotal)}</span>
            </div>
            <div class="summary-row">
                <span>تخفیف:</span>
                <span>${formatCurrency(totalDiscount)}</span>
            </div>
            <div class="summary-row">
                <span>مالیات بر ارزش افزوده (%۹):</span>
                <span>${formatCurrency(tax)}</span>
            </div>
            <div class="summary-row total">
                <span>مبلغ قابل پرداخت:</span>
                <span>${formatCurrency(total)}</span>
            </div>
            <div class="amount-in-words">
                <strong>مبالغ به حروف:</strong> ${totalInWords}
            </div>
        `;
    }
    
    // Download invoice as PNG
    function downloadInvoice() {
        const invoiceElement = document.getElementById('invoicePreview');
        
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال ایجاد...';
        downloadBtn.disabled = true;
        
        html2canvas(invoiceElement, {
            scale: 2,
            useCORS: true,
            backgroundColor: '#ffffff'
        }).then(canvas => {
            const link = document.createElement('a');
            link.download = `فاکتور-${document.getElementById('invoiceNumber').value}.png`;
            link.href = canvas.toDataURL('image/png');
            link.click();
            
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> دانلود فاکتور (PNG)';
            downloadBtn.disabled = false;
        });
    }
    
    // Helper functions
    function formatNumber(num) {
        return new Intl.NumberFormat('fa-IR').format(num);
    }
    
    function formatCurrency(num) {
        return formatNumber(num) + ' ریال';
    }
    
    function getJalaliDate(date) {
        // Simplified - in production use a proper Jalali library
        const jalaliMonths = ['فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور', 'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند'];
        const jalaliYear = 1402 + Math.floor(Math.random() * 2); // Simplified
        const month = jalaliMonths[date.getMonth()];
        const day = date.getDate();
        return `${jalaliYear}/${month}/${day}`;
    }
    
    function getGregorianDate(date) {
        return date.toISOString().split('T')[0].replace(/-/g, '/');
    }
    
    function convertToPersianWords(num) {
        // Simplified version - for production use a proper library
        const units = ['', 'هزار', 'میلیون', 'میلیارد', 'تریلیون'];
        let words = '';
        let temp = Math.floor(num / 1000);
        
        if (temp > 0) {
            words = formatNumber(temp) + ' هزار ';
        }
        
        const remainder = num % 1000;
        if (remainder > 0) {
            words += formatNumber(remainder);
        }
        
        return words + ' تومان';
    }
}
