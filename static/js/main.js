// 전역 변수
let currentSearchResults = null;

// DOM 요소들
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const loadingState = document.getElementById('loadingState');
const searchResults = document.getElementById('searchResults');
const errorMessage = document.getElementById('errorMessage');
const moviesList = document.getElementById('moviesList');
const resultsTitle = document.getElementById('resultsTitle');
const resultsCount = document.getElementById('resultsCount');
const errorText = document.getElementById('errorText');
const retryBtn = document.getElementById('retryBtn');

// 모달 관련
const explanationModal = document.getElementById('explanationModal');
const modalTitle = document.getElementById('modalTitle');
const modalLoading = document.getElementById('modalLoading');
const modalExplanation = document.getElementById('modalExplanation');
const closeModal = document.getElementById('closeModal');

// 이벤트 리스너 등록
document.addEventListener('DOMContentLoaded', function() {
    // 검색 버튼 클릭
    searchBtn.addEventListener('click', handleSearch);
    
    // 엔터키로 검색
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleSearch();
        }
    });
    
    // 재시도 버튼
    retryBtn.addEventListener('click', handleSearch);
    
    // 모달 닫기
    closeModal.addEventListener('click', closeExplanationModal);
    
    // 모달 배경 클릭으로 닫기
    explanationModal.addEventListener('click', function(e) {
        if (e.target === explanationModal) {
            closeExplanationModal();
        }
    });
    
    // ESC 키로 모달 닫기
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
            closeExplanationModal();
        }
    });
    
    // 검색창 포커스
    searchInput.focus();
});

// 검색 처리
async function handleSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showError('검색어를 입력해주세요.');
        return;
    }
    
    // UI 상태 변경
    showLoading();
    
    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || '검색 중 오류가 발생했습니다.');
        }
        
        currentSearchResults = data;
        showResults(data);
        
    } catch (error) {
        console.error('검색 오류:', error);
        showError(error.message);
    }
}

// 로딩 상태 표시
function showLoading() {
    hideAllStates();
    loadingState.classList.remove('hidden');
}

// 검색 결과 표시
function showResults(data) {
    hideAllStates();
    
    const movies = data.movies || [];
    const query = data.query || '';
    
    resultsTitle.textContent = `"${query}" 검색 결과`;
    resultsCount.textContent = `총 ${movies.length}개의 영화를 찾았습니다.`;
    
    // 영화 카드들 생성
    moviesList.innerHTML = '';
    movies.forEach((movie, index) => {
        const movieCard = createMovieCard(movie, index + 1);
        moviesList.appendChild(movieCard);
    });
    
    searchResults.classList.remove('hidden');
}

// 영화 카드 생성
function createMovieCard(movie, rank) {
    const card = document.createElement('div');
    card.className = 'movie-card';
    
    // 줄거리 축약
    const plot = movie.plot || '줄거리 정보가 없습니다.';
    const shortPlot = plot.length > 150 ? plot.substring(0, 150) + '...' : plot;
    
    card.innerHTML = `
        <div class="movie-poster" onclick="goToMovieDetail(${rank})">
            <i class="fas fa-film"></i>
        </div>
        <div class="movie-info">
            <div class="movie-header">
                <div class="movie-title">
                    <h3>${movie.title || 'Unknown Title'}</h3>
                    <div class="movie-year">${movie.year || 'Unknown Year'}</div>
                </div>
                <button class="explain-btn" onclick="showExplanation(${rank})">
                    <i class="fas fa-lightbulb"></i> 설명
                </button>
            </div>
            <div class="movie-plot">${shortPlot}</div>
        </div>
    `;
    
    return card;
}

// 영화 상세 페이지로 이동
function goToMovieDetail(rank) {
    window.location.href = `/movie/${rank}`;
}

// LLM 설명 모달 표시 (수정: JSON 데이터에서 직접 가져오기)
async function showExplanation(rank) {
    // 모달 열기
    explanationModal.classList.remove('hidden');
    modalLoading.classList.remove('hidden');
    modalExplanation.classList.add('hidden');
    
    // 영화 정보 가져오기
    const movie = currentSearchResults.movies[rank - 1];
    modalTitle.textContent = `"${movie.title}" 추천 이유`;
    
    // 검색 결과 JSON에서 직접 설명 가져오기
    let explanation = '설명 정보가 없습니다.';
    
    if (movie.llm_analysis && movie.llm_analysis.reason) {
        explanation = movie.llm_analysis.reason;
    } else {
        console.warn('LLM 분석 데이터가 없습니다:', movie);
    }
    
    showModalExplanation(explanation);
}

// 모달에 설명 표시
function showModalExplanation(explanation) {
    modalLoading.classList.add('hidden');
    modalExplanation.classList.remove('hidden');
    modalExplanation.innerHTML = `<p>${explanation}</p>`;
}

// 모달 닫기
function closeExplanationModal() {
    explanationModal.classList.add('hidden');
}

// 오류 메시지 표시
function showError(message) {
    hideAllStates();
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
}

// 모든 상태 숨기기
function hideAllStates() {
    loadingState.classList.add('hidden');
    searchResults.classList.add('hidden');
    errorMessage.classList.add('hidden');
}

// 유틸리티 함수들
function formatMovieTitle(title, year) {
    return `${title}${year ? ` (${year})` : ''}`;
}

function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// 검색어 하이라이트 (미래 기능)
function highlightSearchTerms(text, query) {
    if (!query) return text;
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<mark>$1</mark>');
}

// 페이지 새로고침 시 포커스 복원
window.addEventListener('load', function() {
    searchInput.focus();
});