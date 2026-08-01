// ============ LIGHTBOX ============
const lightbox = document.getElementById('lightbox');
const lightboxImg = lightbox.querySelector('img');

// Attach click listeners to all skill tree images
document.querySelectorAll('.skill-tree-item img').forEach(img => {
    img.addEventListener('click', () => {
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt;
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    });
});

function closeLightbox() {
    lightbox.classList.remove('active');
    document.body.style.overflow = '';
}

// Close on backdrop click
lightbox.addEventListener('click', (e) => {
    if (e.target === lightbox) {
        closeLightbox();
    }
});

// Close on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox.classList.contains('active')) {
        closeLightbox();
    }
});

// ============ ACCORDION TOGGLE ============
function toggleCard(header) {
    const card = header.closest('.build-card');
    card.classList.toggle('open');
}