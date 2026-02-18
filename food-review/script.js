// Get the form and reviews container
const reviewForm = document.getElementById('reviewForm');
const reviewsContainer = document.getElementById('reviewsContainer');

// Function to get star display based on rating
function getStars(rating) {
    const fullStars = '★'.repeat(rating);
    const emptyStars = '☆'.repeat(5 - rating);
    return fullStars + emptyStars;
}

// Function to get current date
function getCurrentDate() {
    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return new Date().toLocaleDateString('en-US', options);
}

// Function to create a review card
function createReviewCard(dishName, restaurant, rating, reviewText, reviewerName, imageUrl) {
    const reviewCard = document.createElement('div');
    reviewCard.className = 'review-card';
    
    // Use default image if none provided
    const imgContent = imageUrl 
        ? `<img src="${imageUrl}" alt="${dishName}" class="food-image">`
        : `<div class="food-image placeholder-default"><span class="placeholder-text">🍽️ ${dishName}</span></div>`;
    
    reviewCard.innerHTML = `
        <div class="review-header">
            ${imgContent}
            <div class="rating">
                <span class="stars">${getStars(parseInt(rating))}</span>
                <span class="rating-number">${rating}/5</span>
            </div>
        </div>
        <div class="review-content">
            <h3>${dishName}</h3>
            <p class="restaurant">🏪 Restaurant: ${restaurant}</p>
            <p class="review-text">${reviewText}</p>
            <div class="review-meta">
                <span class="reviewer">Reviewed by: ${reviewerName}</span>
                <span class="date">${getCurrentDate()}</span>
            </div>
        </div>
    `;
    
    return reviewCard;
}

// Handle form submission
reviewForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    // Get form values
    const dishName = document.getElementById('dishName').value;
    const restaurant = document.getElementById('restaurant').value;
    const rating = document.getElementById('rating').value;
    const reviewText = document.getElementById('reviewText').value;
    const reviewerName = document.getElementById('reviewerName').value;
    const imageUrl = document.getElementById('imageUrl').value;
    
    // Create and add the new review card
    const newReview = createReviewCard(dishName, restaurant, rating, reviewText, reviewerName, imageUrl);
    reviewsContainer.insertBefore(newReview, reviewsContainer.firstChild);
    
    // Reset the form
    reviewForm.reset();
    
    // Show success message
    alert('Review submitted successfully!');
    
    // Scroll to the reviews section
    document.getElementById('reviews').scrollIntoView({ behavior: 'smooth' });
    
    // Add animation to the new card
    newReview.style.animation = 'slideIn 0.5s ease-out';
});

// Add CSS animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({ behavior: 'smooth' });
        }
    });
});
