// فاکتورساز حرفه‌ای - جاوااسکریپت کامل
document.addEventListener('DOMContentLoaded', function() {
    // ===== GLOBAL VARIABLES =====
    const config = {
        template: 'modern',
        color: '#4361ee',
        dateFormat: 'jalali',
        fontSize: 16
    };
    
    const MAX_AMOUNT = 20000000; // 20 میلیون ریال
    
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
    
    let previewCounter = 1;
    let touchStartX = 0;
    let touchStartY = 0;
    
    // ===== DOM ELEMENTS =====
    const elements = {
        // Panels
        controlsPanel: document.getElementById('controlsPanel'),
        dataPanel: document.getElementById('dataPanel'),
        previewPanel: document.getElementById('previewPanel'),
        
        // Mobile Tabs
        mobileTabs: document.querySelectorAll('.mobile-tab'),
        
        // Invoice Preview
        invoicePreview: document.getElementById('invoicePreview'),
        previewCount: document.getElementById('previewCount'),
        
        // Input Fields
        sellerName: document.getElementById('sellerName'),
        sellerPhone: document.getElementById('sellerPhone'),
        sellerAddress: document.getElementById('sellerAddress'),
        buyerName: document.getElementById('buyerName'),
        buyerPhone: document.getElementById('buyerPhone'),
        invoiceNumber: document.getElementById('invoiceNumber'),
        customColor: document.getElementById('customColor'),
        fontSize: document.getElementById('fontSize'),
        currentFontSize: document.getElementById('currentFontSize'),
        
        // Items List
        itemsList: document.getElementById('itemsList'),
        
        // Totals
        subtotal: document.getElementById('subtotal'),
        discountAmount: document.getElementById('discountAmount'),
        taxAmount: document.getElementById('taxAmount'),
        totalAmount: document.getElementById('totalAmount'),
        
        // Buttons
        addItemBtn: document.getElementById('addItemBtn'),
        downloadBtn: document.getElementById('downloadBtn'),
        resetBtn: document.getElementById('resetBtn'),
        printBtn: document.getElementById('printBtn'),
        shareBtn: document.getElementById('shareBtn'),
        
        // Menu
        menuToggle: document.getElementById('menuToggle'),
        mobileMenu: document.getElementById('mobileMenu'),
        menuClose: document.querySelector('.menu-close'),
        
        // Modals
        helpModal: document.getElementById('helpModal'),
        helpLink: document.getElementById('helpLink'),
        contactLink: document.getElementById('contactLink')
    };
    
    // ===== INITIALIZATION =====
    init();
    
    function init() {
        setupTouchEvents();
        setupEventListeners();
        renderInvoice();
        renderItems();
        updateTotals();
        
        // Set initial panel visibility for mobile
        updateMobilePanelVisibility();
        
        // Load html2canvas if not loaded
        if (typeof html2canvas === 'undefined') {
            loadHtml2Canvas();
        }
    }
    
    // ===== TOUCH EVENTS =====
    function setupTouchEvents() {
        // Prevent zoom
        document.addEventListener('touchmove', function(event) {
            if (event.touches.length > 1) {
                event.preventDefault();
            }
        }, { passive: false });
        
        // Swipe detection for mobile panels
        document.addEventListener('touchstart', handleTouchStart, { passive: true });
        document.addEventListener('touchend', handleTouchEnd, { passive: true });
        
        // Smooth scrolling for touch devices
        document.querySelectorAll('.scrollable').forEach(element => {
            element.addEventListener('touchstart', function() {
                this.style.scrollBehavior = 'auto';
            });
            
            element.addEventListener('touchend', function() {
                this.style.scrollBehavior = 'smooth';
            });
        });
    }
    
    function handleTouchStart(event) {
        touchStartX = event.touches[0].clientX;
        touchStartY = event.touches[0].clientY;
    }
    
    function handleTouchEnd(event) {
        if (!touchStartX || !touchStartY) return;
        
        const touchEndX = event.changedTouches[0].clientX;
        const touchEndY = event.changedTouches[0].clientY;
        
        const diffX = touchStartX - touchEndX;
        const diffY = touchStartY - touchEndY;
        
        // Only consider horizontal swipes
        if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 50) {
            if (diffX > 0) {
                // Swipe left - next panel
                nextMobilePanel();
            } else {
                // Swipe right - previous panel
                prevMobilePanel();
            }
        }
        
        touchStartX = 0;
        touchStartY = 0;
    }
    
    // ===== MOBILE PANELS =====
    function updateMobilePanelVisibility() {
        if (window.innerWidth > 992) return;
        
        const activeTab = document.querySelector('.mobile-tab.active').dataset.tab;
        
        // Hide all panels
        if (elements.controlsPanel) elements.controlsPanel.classList.remove('active');
        if (elements.dataPanel) elements.dataPanel.classList.remove('active');
        if (elements.previewPanel) elements.previewPanel.classList.remove('active');
        
        // Show active panel
        switch(activeTab) {
            case 'controls':
                if (elements.controlsPanel) elements.controlsPanel.classList.add('active');
                break;
            case 'data':
                if (elements.dataPanel) elements.dataPanel.classList.add('active');
                break;
            case 'preview':
                if (elements.previewPanel) elements.previewPanel.classList.add('active');
                break;
        }
    }
    
    function nextMobilePanel() {
        const tabs = Array.from(document.querySelectorAll('.mobile-tab'));
        const currentIndex = tabs.findIndex(tab => tab.classList.contains('active'));
        
        if (currentIndex < tabs.length - 1) {
            tabs[currentIndex].classList.remove('active');
            tabs[currentIndex + 1].classList.add('active');
            tabs[currentIndex + 1].click();
            updateMobilePanelVisibility();
        }
    }
    
    function prevMobilePanel() {
        const tabs = Array.from(document.querySelectorAll('.mobile-tab'));
        const currentIndex = tabs.findIndex(tab => tab.classList.contains('active'));
        
        if (currentIndex > 0) {
            tabs[currentIndex].classList.remove('active');
            tabs[currentIndex - 1].classList.add('active');
            tabs[currentIndex - 1].click();
            updateMobilePanelVisibility();
        }
    }
    
    // ===== EVENT LISTENERS =====
    function setupEventListeners() {
        // Mobile Tabs
        elements.mobileTabs.forEach(tab => {
            tab.addEventListener('click', function() {
                elements.mobileTabs.forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                updateMobilePanelVisibility();
            });
        });
        
        // Desktop Tabs
        document.querySelectorAll('.desktop-tab').forEach(tab => {
            tab.addEventListener('click', function() {
                const tabId = this.dataset.tab;
                
                // Update active tab
                document.querySelectorAll('.desktop-tab').forEach(t => t.classList.remove('active'));
                this.classList.add('active');
                
                // Show corresponding content
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.classList.remove('active');
                });
                document.getElementById(tabId + 'Tab').classList.add('active');
            });
        });
        
        // Template Selection
        document.querySelectorAll('.template-card').forEach(card => {
            card.addEventListener('click', function() {
                document.querySelectorAll('.template-card').forEach(c => c.classList.remove('active'));
                this.classList.add('active');
                config.template = this.dataset.template;
                renderInvoice();
            });
        });
        
        // Color Selection
        document.querySelectorAll('.color-option').forEach(option => {
            option.addEventListener('click', function() {
                document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
                this.classList.add('active');
                config.color = this.dataset.color;
                elements.customColor.value = config.color;
                renderInvoice();
            });
        });
        
        // Custom Color
        elements.customColor.addEventListener('input', function(e) {
            config.color = e.target.value;
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            renderInvoice();
        });
        
        // Date Format
        document.querySelectorAll('input[name="dateFormat"]').forEach(radio => {
            radio.addEventListener('change', function(e) {
                config.dateFormat = e.target.value;
                renderInvoice();
            });
        });
        
        // Font Size
        elements.fontSize.addEventListener('input', function(e) {
            config.fontSize = parseInt(e.target.value);
            if (elements.currentFontSize) {
                elements.currentFontSize.textContent = config.fontSize;
            }
            renderInvoice();
        });
        
        // Input Fields
        ['sellerName', 'sellerPhone', 'sellerAddress', 'buyerName', 'buyerPhone', 'invoiceNumber']
            .forEach(id => {
                const element = document.getElementById(id);
                if (element) {
                    element.addEventListener('input', function() {
                        renderInvoice();
                        validateAmounts();
                    });
                    
                    // Touch optimization
                    element.addEventListener('touchstart', function() {
                        this.style.fontSize = '16px'; // Prevent zoom on iOS
                    });
                }
            });
        
        // Add Item
        if (elements.addItemBtn) {
            elements.addItemBtn.addEventListener('click', addNewItem);
        }
        
        // Download
        if (elements.downloadBtn) {
            elements.downloadBtn.addEventListener('click', downloadInvoice);
        }
        
        // Reset
        if (elements.resetBtn) {
            elements.resetBtn.addEventListener('click', resetToDefaults);
        }
        
        // Print
        if (elements.printBtn) {
            elements.printBtn.addEventListener('click', printInvoice);
        }
        
        // Share
        if (elements.shareBtn) {
            elements.shareBtn.addEventListener('click', shareInvoice);
        }
        
        // Mobile Menu
        if (elements.menuToggle && elements.mobileMenu && elements.menuClose) {
            elements.menuToggle.addEventListener('click', function() {
                elements.mobileMenu.classList.add('active');
            });
            
            elements.menuClose.addEventListener('click', function() {
                elements.mobileMenu.classList.remove('active');
            });
            
            // Close menu on outside click
            elements.mobileMenu.addEventListener('click', function(e) {
                if (e.target === this) {
                    this.classList.remove('active');
                }
            });
        }
        
        // Help Modal
        if (elements.helpModal) {
            const modalClose = elements.helpModal.querySelector('.modal-close');
            const modalOverlay = elements.helpModal.querySelector('.modal-overlay');
            
            // Open modal
            const openModal = function() {
                elements.helpModal.classList.add('active');
            };
            
            if (elements.helpLink) elements.helpLink.addEventListener('click', function(e) {
                e.preventDefault();
                openModal();
            });
            
            // Mobile help
            const mobileHelp = document.getElementById('mobileHelp');
            if (mobileHelp) mobileHelp.addEventListener('click', function(e) {
                e.preventDefault();
                openModal();
            });
            
            // Close modal
            if (modalClose) {
                modalClose.addEventListener('click', function() {
                    elements.helpModal.classList.remove('active');
                });
            }
            
            if (modalOverlay) {
                modalOverlay.addEventListener('click', function() {
                    elements.helpModal.classList.remove('active');
                });
            }
        }
        
        // Contact Link
        if (elements.contactLink) {
            elements.contactLink.addEventListener('click', function(e) {
                e.preventDefault();
                showNotification('صفحه تماس با ما به زودی اضافه می‌شود', 'info');
            });
        }
        
        // Mobile contact
        const mobileContact = document.getElementById('mobileContact');
        if (mobileContact) {
            mobileContact.addEventListener('click', function(e) {
                e.preventDefault();
                showNotification('صفحه تماس با ما به زودی اضافه می‌شود', 'info');
            });
        }
        
        // Window resize
        window.addEventListener('resize', function() {
            updateMobilePanelVisibility();
        });
    }
    
    // ===== INVOICE RENDERING =====
    function renderInvoice() {
        const data = getInvoiceData();
        const html = generateInvoiceHTML(data);
        
        if (elements.invoicePreview) {
            elements.invoicePreview.innerHTML = html;
            
            // Update preview counter
            previewCounter++;
            if (elements.previewCount) {
                elements.previewCount.textContent = previewCounter;
            }
            
            // Trigger animation
            elements.invoicePreview.style.opacity = '0.8';
            setTimeout(() => {
                elements.invoicePreview.style.opacity = '1';
            }, 50);
        }
    }
    
    function getInvoiceData() {
        return {
            seller: {
                name: elements.sellerName?.value || 'فروشگاه موبایل پارسیان',
                phone: elements.sellerPhone?.value || '021-12345678',
                address: elements.sellerAddress?.value || 'تهران، خیابان ولیعصر، پلاک ۲۴۰'
            },
            buyer: {
                name: elements.buyerName?.value || 'سعید کنانی',
                phone: elements.buyerPhone?.value || '09137657870'
            },
            invoiceNumber: elements.invoiceNumber?.value || 'INV-1403-001',
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
        
        // Apply limit
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
        
        // Generate items table rows
        let itemsHTML = '';
        items.forEach((item, index) => {
            const total = item.quantity * item.unitPrice;
            const afterDiscount = total * (1 - item.discount/100);
            const rowStyle = index % 2 === 0 ? 'background: #fafafa;' : '';
            
            itemsHTML += `
                <tr style="${rowStyle} border-bottom: 1px solid #eee;">
                    <td style="padding: 12px 15px; text-align: center;">${index + 1}</td>
                    <td style="padding: 12px 15px;">${item.description}</td>
                    <td style="padding: 12px 15px; text-align: center;">${item.quantity}</td>
                    <td style="padding: 12px 15px; text-align: left; direction: ltr;">${formatNumber(item.unitPrice)}</td>
                    <td style="padding: 12px 15px; text-align: center;">${item.discount}%</td>
                    <td style="padding: 12px 15px; text-align: left; direction: ltr;">${formatNumber(afterDiscount)}</td>
                </tr>
            `;
        });
        
        return `
            <div class="invoice" style="font-size: ${config.fontSize}px; color: #333; font-family: 'Vazirmatn', sans-serif;">
                <!-- Header -->
                <div style="border-bottom: 3px solid ${config.color}; padding-bottom: 20px; margin-bottom: 30px;">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
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
                        <div style="opacity: 0.1; font-size: 3em; font-weight: 900; transform: rotate(-15deg); user-select: none; max-width: 200px; word-break: break-word;">
                            ${seller.name}
                        </div>
                    </div>
                </div>
                
                <!-- Parties -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 40px;">
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; border-right: 5px solid ${config.color};">
                        <h3 style="color: ${config.color}; margin-bottom: 15px; font-size: 1.2em; font-weight: 700;">فروشنده</h3>
                        <div style="line-height: 1.8;">
                            <p style="margin-bottom: 8px;">
                                <i class="fas fa-store" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${seller.name}
                            </p>
                            <p style="margin-bottom: 8px;">
                                <i class="fas fa-phone" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${seller.phone}
                            </p>
                            <p>
                                <i class="fas fa-map-marker-alt" style="color: ${config.color}; margin-left: 8px;"></i> 
                                ${seller.address}
                            </p>
                        </div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; border-right: 5px solid ${config.color};">
                        <h3 style="color: ${config.color}; margin-bottom: 15px; font-size: 1.2em; font-weight: 700;">خریدار</h3>
                        <div style="line-height: 1.8;">
                            <p style="margin-bottom: 8px;">
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
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; margin: 30px 0; min-width: 600px;">
                        <thead>
                            <tr style="background: ${config.color}; color: white;">
                                <th style="padding: 15px; text-align: center; font-weight: 700;">ردیف</th>
                                <th style="padding: 15px; text-align: right; font-weight: 700;">شرح کالا / خدمت</th>
                                <th style="padding: 15px; text-align: center; font-weight: 700;">تعداد</th>
                                <th style="padding: 15px; text-align: center; font-weight: 700;">قیمت واحد (ریال)</th>
                                <th style="padding: 15px; text-align: center; font-weight: 700;">تخفیف %</th>
                                <th style="padding: 15px; text-align: center; font-weight: 700;">مجموع (ریال)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${itemsHTML}
                        </tbody>
                    </table>
                </div>
                
                <!-- Summary -->
                <div style="background: #f8f9fa; padding: 25px; border-radius: 12px; margin: 40px 0; border: 2px dashed ${config.color}40;">
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>جمع کل:</span>
                        <span style="font-weight: 600;">${formatCurrency(totals.subtotal)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>تخفیف:</span>
                        <span style="font-weight: 600;">${formatCurrency(totals.discount)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #ddd;">
                        <span>مالیات بر ارزش افزوده (۹٪):</span>
                        <span style="font-weight: 600;">${formatCurrency(totals.tax)}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 15px 0; font-size: 1.3em; font-weight: 900; border-top: 2px solid ${config.color}; margin-top: 10px;">
                        <span>مبلغ قابل پرداخت:</span>
                        <span style="color: ${config.color};">${formatCurrency(totals.total)}</span>
                    </div>
                    
                    <div style="background: white; padding: 15px; border-radius: 8px; margin-top: 20px; border-right: 4px solid ${config.color}; line-height: 1.6;">
                        <strong>مبلغ به حروف:</strong> ${totals.totalInWords}
                    </div>
                </div>
                
                <!-- Footer -->
                <div style="margin-top: 50px; padding-top: 30px; border-top: 2px solid #eee;">
                    <div style="margin-bottom: 30px; line-height: 1.6;">
                        <p><strong>توضیحات:</strong> کالاها ظرف ۳ روز کاری آماده تحویل می‌باشد. گارانتی ۱۸ ماهه</p>
                        <p><strong>روش پرداخت:</strong> نقدی</p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px; margin: 40px 0;">
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
                    
                    <div style="text-align: center; color: #666; font-style: italic; padding: 20px; background: #f8f9fa; border-radius: 8px; line-height: 1.6;">
                        این فاکتور به صورت الکترونیکی صادر شده و نیاز به مهر و امضای فیزیکی ندارد
                    </div>
                </div>
            </div>
        `;
    }
    
    // ===== ITEMS MANAGEMENT =====
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
                       data-index="${index}" data-field="quantity" placeholder="1">
                <input type="number" class="item-price" value="${item.unitPrice}" 
                       data-index="${index}" data-field="unitPrice" placeholder="قیمت">
                <input type="number" class="item-discount" value="${item.discount}" min="0" max="100"
                       data-index="${index}" data-field="discount" placeholder="0">
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
        // Item inputs
        document.querySelectorAll('.item-desc, .item-qty, .item-price, .item-discount').forEach(input => {
            input.addEventListener('input', function(e) {
                const index = parseInt(this.dataset.index);
                const field = this.dataset.field;
                let value = this.type === 'number' ? parseFloat(this.value) || 0 : this.value;
                
                // Validate amount limit
                if ((field === 'unitPrice' || field === 'quantity') && value > MAX_AMOUNT) {
                    value = MAX_AMOUNT;
                    this.value = value;
                    showNotification(`مبلغ نمی‌تواند بیشتر از ${formatNumber(MAX_AMOUNT)} ریال باشد`, 'error');
                }
                
                items[index][field] = value;
                renderInvoice();
                updateTotals();
                validateAmounts();
                
                // Update row total
                if (field !== 'description') {
                    const item = items[index];
                    const total = item.quantity * item.unitPrice * (1 - item.discount/100);
                    this.closest('.item-row').querySelector('.item-total').textContent = formatCurrency(total);
                }
            });
        });
        
        // Remove buttons
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
    
    // ===== VALIDATION =====
    function validateAmounts() {
        let totalAmount = 0;
        
        items.forEach(item => {
            const itemTotal = item.quantity * item.unitPrice;
            totalAmount += itemTotal;
        });
        
        if (totalAmount > MAX_AMOUNT) {
            showNotification(`مجموع مبلغ فاکتور از ${formatNumber(MAX_AMOUNT)} ریال بیشتر شد!`, 'error');
            return false;
        }
        
        return true;
    }
    
    // ===== TOTALS =====
    function updateTotals() {
        const totals = calculateTotals();
        
        if (elements.subtotal) elements.subtotal.textContent = formatCurrency(totals.subtotal);
        if (elements.discountAmount) elements.discountAmount.textContent = formatCurrency(totals.discount);
        if (elements.taxAmount) elements.taxAmount.textContent = formatCurrency(totals.tax);
        if (elements.totalAmount) elements.totalAmount.textContent = formatCurrency(totals.total);
    }
    
    // ===== DOWNLOAD =====
    function downloadInvoice() {
        // Validate before download
        if (!validateAmounts()) {
            return;
        }
        
        if (!elements.downloadBtn) return;
        
        const btn = elements.downloadBtn;
        const originalHTML = btn.innerHTML;
        
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> در حال ایجاد...';
        btn.disabled = true;
        
        // Check html2canvas
        if (typeof html2canvas === 'undefined') {
            showNotification('لطفاً چند لحظه صبر کنید...', 'info');
            setTimeout(() => {
                if (typeof html2canvas !== 'undefined') {
                    downloadInvoice();
                } else {
                    showNotification('خطا در بارگذاری ابزار ایجاد تصویر', 'error');
                    btn.innerHTML = originalHTML;
                    btn.disabled = false;
                }
            }, 1000);
            return;
        }
        
        // Wait for fonts and render
        setTimeout(() => {
            document.fonts.ready.then(() => {
                const invoiceElement = elements.invoicePreview.querySelector('.invoice') || elements.invoicePreview;
                
                const options = {
                    scale: 2,
                    useCORS: true,
                    allowTaint: true,
                    backgroundColor: '#ffffff',
                    logging: false,
                    onclone: function(clonedDoc) {
                        // Ensure fonts are loaded
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
                
                html2canvas(invoiceElement, options)
                    .then(canvas => {
                        const link = document.createElement('a');
                        const invoiceNum = elements.invoiceNumber?.value || 'جدید';
                        link.download = `فاکتور-${invoiceNum}.png`;
                        link.href = canvas.toDataURL('image/png', 1.0);
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
            });
        }, 500);
    }
    
    // ===== OTHER ACTIONS =====
    function printInvoice() {
        window.print();
    }
    
    function shareInvoice() {
        if (navigator.share) {
            navigator.share({
                title: 'فاکتور حرفه‌ای',
                text: 'فاکتور ایجاد شده با فاکتورساز حرفه‌ای',
                url: window.location.href
            });
        } else {
            showNotification('امکان اشتراک‌گذاری در این مرورگر وجود ندارد', 'info');
        }
    }
    
    function resetToDefaults() {
        if (confirm('آیا از بازنشانی همه تنظیمات به حالت اولیه اطمینان دارید؟')) {
            // Reset config
            config.template = 'modern';
            config.color = '#4361ee';
            config.dateFormat = 'jalali';
            config.fontSize = 16;
            
            // Reset items
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
            
            // Reset UI elements
            document.querySelectorAll('.template-card').forEach(o => o.classList.remove('active'));
            document.querySelector('.template-card[data-template="modern"]').classList.add('active');
            
            document.querySelectorAll('.color-option').forEach(o => o.classList.remove('active'));
            document.querySelector('.color-option[data-color="#4361ee"]').classList.add('active');
            
            if (elements.customColor) elements.customColor.value = '#4361ee';
            
            document.querySelector('input[name="dateFormat"][value="jalali"]').checked = true;
            
            if (elements.fontSize) elements.fontSize.value = 16;
            if (elements.currentFontSize) elements.currentFontSize.textContent = '16';
            
            // Reset input fields
            if (elements.sellerName) elements.sellerName.value = 'فروشگاه موبایل پارسیان';
            if (elements.sellerPhone) elements.sellerPhone.value = '021-12345678';
            if (elements.sellerAddress) elements.sellerAddress.value = 'تهران، خیابان ولیعصر، پلاک ۲۴۰';
            if (elements.buyerName) elements.buyerName.value = 'سعید کنانی';
            if (elements.buyerPhone) elements.buyerPhone.value = '09137657870';
            if (elements.invoiceNumber) elements.invoiceNumber.value = 'INV-1403-001';
            
            // Reset desktop tabs
            document.querySelectorAll('.desktop-tab').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.desktop-tab[data-tab="seller"]').classList.add('active');
            
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById('sellerTab').classList.add('active');
            
            // Reset mobile tabs
            document.querySelectorAll('.mobile-tab').forEach(btn => btn.classList.remove('active'));
            document.querySelector('.mobile-tab[data-tab="controls"]').classList.add('active');
            updateMobilePanelVisibility();
            
            // Re-render
            renderInvoice();
            renderItems();
            updateTotals();
            
            showNotification('همه تنظیمات بازنشانی شد', 'success');
        }
    }
    
    // ===== UTILITY FUNCTIONS =====
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
        let notification = document.getElementById('notification');
        
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'notification';
            notification.innerHTML = `
                <div class="notification-content">
                    <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
                    <span id="notificationText">${message}</span>
                </div>
            `;
            document.body.appendChild(notification);
        }
        
        const icon = notification.querySelector('i');
        const text = notification.querySelector('#notificationText');
        
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
    
    function loadHtml2Canvas() {
        const script = document.createElement('script');
        script.src = 'https://html2canvas.hertzen.com/dist/html2canvas.min.js';
        script.onload = function() {
            console.log('html2canvas loaded successfully');
        };
        script.onerror = function() {
            console.error('Failed to load html2canvas');
        };
        document.head.appendChild(script);
    }
});
