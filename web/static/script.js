// 페이지 로드 시 애니메이션 효과
document.addEventListener('DOMContentLoaded', function() {
    // 로고 클릭 효과
    const logo = document.querySelector('.logo svg');
    logo.addEventListener('click', function() {
        this.style.transform = 'scale(1.1) rotate(360deg)';
        setTimeout(() => {
            this.style.transform = 'scale(1) rotate(0deg)';
        }, 600);
    });

    // 기능 카드들에 순차적 애니메이션 효과 추가
    const featureCards = document.querySelectorAll('.feature-card');
    featureCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.6s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 200 + (index * 150));
    });

    // 배경 클릭 시 파티클 효과
    document.addEventListener('click', createParticle);
});

// 검색 시작 버튼 클릭 이벤트
function startSearch() {
    const button = document.querySelector('.cta-button');
    
    // 버튼 클릭 애니메이션
    button.style.transform = 'scale(0.95)';
    setTimeout(() => {
        button.style.transform = 'scale(1)';
    }, 150);

    // 로딩 상태 표시
    const originalText = button.innerHTML;
    button.innerHTML = `
        <div class="loading-spinner"></div>
        로딩중...
    `;
    button.disabled = true;

    // CSS로 스피너 스타일 추가
    if (!document.querySelector('#spinner-style')) {
        const style = document.createElement('style');
        style.id = 'spinner-style';
        style.textContent = `
            .loading-spinner {
                width: 16px;
                height: 16px;
                border: 2px solid rgba(255,255,255,0.3);
                border-top: 2px solid white;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }

    // welcome.html로 즉시 이동 (테스트용)
    setTimeout(() => {
        // welcome.html로 페이지 이동
        window.location.href = 'welcome.html';
    }, 500);
}

// 클릭 시 파티클 효과 생성
function createParticle(e) {
    const particle = document.createElement('div');
    particle.style.position = 'fixed';
    particle.style.left = e.clientX + 'px';
    particle.style.top = e.clientY + 'px';
    particle.style.width = '6px';
    particle.style.height = '6px';
    particle.style.background = 'rgba(255, 255, 255, 0.8)';
    particle.style.borderRadius = '50%';
    particle.style.pointerEvents = 'none';
    particle.style.zIndex = '1000';
    particle.style.transform = 'translate(-50%, -50%)';
    
    document.body.appendChild(particle);
    
    // 파티클 애니메이션
    particle.animate([
        { transform: 'translate(-50%, -50%) scale(0)', opacity: 1 },
        { transform: 'translate(-50%, -50%) scale(1)', opacity: 0.8 },
        { transform: 'translate(-50%, -50%) scale(0)', opacity: 0 }
    ], {
        duration: 600,
        easing: 'ease-out'
    }).addEventListener('finish', () => {
        particle.remove();
    });
}

// 키보드 단축키 (Enter 키로 검색 시작)
document.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        startSearch();
    }
});

// 스크롤 허용으로 변경
// document.body.style.overflow = 'hidden';

// 마우스 이동에 따른 배경 그라디언트 효과
document.addEventListener('mousemove', function(e) {
    const x = e.clientX / window.innerWidth;
    const y = e.clientY / window.innerHeight;
    
    document.body.style.background = `
        linear-gradient(
            ${135 + (x * 30)}deg, 
            hsl(${230 + (x * 20)}, ${70 + (y * 10)}%, ${65 + (x * 5)}%) 0%, 
            hsl(${280 + (y * 20)}, ${60 + (x * 15)}%, ${55 + (y * 10)}%) 100%
        )
    `;
});

// 페이지 가시성 변경 시 애니메이션 일시정지/재개
document.addEventListener('visibilitychange', function() {
    const floatingElements = document.querySelectorAll('.floating-element');
    const logo = document.querySelector('.logo svg');
    
    if (document.hidden) {
        floatingElements.forEach(el => el.style.animationPlayState = 'paused');
        logo.style.animationPlayState = 'paused';
    } else {
        floatingElements.forEach(el => el.style.animationPlayState = 'running');
        logo.style.animationPlayState = 'running';
    }
});
