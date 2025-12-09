// 전역 변수
let currentSearchResults = null;
let streamingResults = []; // 스트리밍 결과 저장
let eventSource = null; // 스트리밍 연결

// 세션 스토리지 키
const SEARCH_STATE_KEY = 'movieSearchState';

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

// 검색 처리 (빠른 검색 우선)
async function handleSearch() {
    const query = searchInput.value.trim();
    
    if (!query) {
        showError('검색어를 입력해주세요.');
        return;
    }
    
    // UI 상태 변경
    showLoading();
    streamingResults = []; // 결과 초기화
    
    console.log('🔍 검색 시작:', query);
    
    try {
        // 1차: 실시간 스트리밍 검색 시도
        console.log('🌊 실시간 스트리밍 검색 시도...');
        await handleStreamingSearch(query);
    } catch (streamingError) {
        console.error('스트리밍 검색 실패, 빠른 검색으로 폴백:', streamingError);
        
        try {
            // 2차: 빠른 검색 (폴백)
            console.log('⚡ 빠른 검색 시도...');
            await handleFastSearch(query);
        } catch (fastError) {
            console.error('빠른 검색 실패, 일반 검색으로 폴백:', fastError);
            
            try {
                // 3차: 일반 검색 (최종 폴백)
                console.log('🔄 일반 검색 시도...');
                await handleRegularSearch(query);
            } catch (regularError) {
                console.error('모든 검색 방법 실패:', regularError);
                showError('검색에 실패했습니다. 서버 상태를 확인해주세요.');
            }
        }
    }
}

// 실시간 스트리밍 검색
async function handleStreamingSearch(query) {
    console.log('🎬 실시간 스트리밍 검색 시작:', query);
    
    // 이전 스트리밍 연결 종료
    if (eventSource) {
        eventSource.close();
    }
    
    // 결과 초기화
    streamingResults = [];
    showResults('스트리밍 검색 시작...', []);
    
    return new Promise((resolve, reject) => {
        try {
            // 스트리밍 검색 요청
            fetch('/streaming-search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('스트리밍 검색 요청 실패');
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                
                function readStream() {
                    return reader.read().then(({ done, value }) => {
                        if (done) {
                            console.log('🎉 스트리밍 검색 완료');
                            resolve();
                            return;
                        }
                        
                        const chunk = decoder.decode(value);
                        const lines = chunk.split('\n');
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.substring(6));
                                    handleStreamingMessage(data);
                                } catch (parseError) {
                                    console.warn('스트리밍 데이터 파싱 오류:', parseError);
                                }
                            }
                        }
                        
                        return readStream();
                    });
                }
                
                readStream().catch(reject);
            })
            .catch(reject);
            
        } catch (error) {
            reject(error);
        }
    });
}

// 스트리밍 메시지 처리 (실시간 LLM 분석)
function handleStreamingMessage(data) {
    console.log('📡 스트리밍 메시지:', data);
    
    switch (data.status) {
        case 'searching':
        case 'loading':
            updateLoadingMessage(data.message);
            break;
            
        case 'llm_start':
            updateLoadingMessage(`🤖 ${data.message}`);
            showResults(`"${searchInput.value.trim()}" 검색 결과`, []);
            break;
            
        case 'analyzing':
            updateLoadingMessage(`🔍 [${data.current}/${data.total}] ${data.message}`);
            break;
            
        case 'error':
            console.error('스트리밍 오류:', data.message);
            updateLoadingMessage(`❌ ${data.message}`);
            break;
            
        case 'completed':
            hideLoading();
            const query = searchInput.value.trim();
            
            // currentSearchResults 업데이트
            currentSearchResults = {
                query,
                movies: streamingResults,
                llm_filtered: true,
                total_count: streamingResults.length
            };
            
            // 올바른 형식으로 결과 표시 (객체로 전달)
            showResults({
                query: data.approved_count === 0 
                    ? `"${query}" 검색 결과` 
                    : `"${query}" 검색 결과 (${data.approved_count}개)`,
                movies: streamingResults,
                llm_filtered: true,
                total_count: streamingResults.length,
                message: data.approved_count === 0 ? '검색 결과가 없습니다.' : null
            });
            
            // 최종 상태 저장
            saveSearchState();
            
            console.log('🎉 실시간 LLM 분석 완료:', data.approved_count, '개 승인');
            break;
    }
    
    // ✨ 실시간으로 승인된 영화 표시!
    if (data.type === 'approved_movie') {
        streamingResults.push(data.data);
        console.log('✅ 실시간 승인 영화:', data.data.title, `(${data.approved_count}번째)`);
        
        // 즉시 화면에 표시 (totalCount 파라미터 추가)
        addMovieToResults(data.data, data.approved_count || streamingResults.length);
        
        // 결과 카운트 실시간 업데이트
        updateResultsCount(streamingResults.length, false);
        updateResultsCount(data.approved_count);
        
        // 상태 저장 (포스터 클릭 후 돌아와도 결과 유지)
        saveSearchState();
        
        // 자동 스크롤은 기본값 off (사용자가 위쪽 영화를 읽고 있을 때 방해하지 않음)
        const autoScrollEnabled = false;
        if (autoScrollEnabled) {
            // 스크롤을 새 영화로 부드럽게 이동
            setTimeout(() => {
                const newMovie = document.querySelector('.movie-card:last-child');
                if (newMovie) {
                    newMovie.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    // 새 영화 하이라이트 효과
                    newMovie.classList.add('newly-added');
                    setTimeout(() => {
                        newMovie.classList.remove('newly-added');
                    }, 2000);
                }
            }, 100);
        }
    }
}

// 로딩 메시지 업데이트
function updateLoadingMessage(message) {
    const loadingText = document.querySelector('#loadingState .loading-text');
    if (loadingText) {
        loadingText.textContent = message;
    }
}

// 결과 카운트 업데이트
function updateResultsCount(count) {
    resultsCount.textContent = count;
    resultsTitle.textContent = count > 0 ? '검색 결과' : '검색 중...';
}

// 개별 영화를 결과에 즉시 추가
function addMovieToResults(movie, totalCount) {
    // rank 결정: totalCount 우선, 없으면 배열 길이 사용
    const rank = totalCount || streamingResults.length;
    
    // 첫 번째 결과일 때 결과 영역 표시
    if (rank === 1) {
        showResults({
            query: searchInput.value.trim(),
            movies: []  // 빈 배열로 초기화
        });
    }
    
    const movieCard = createMovieCard(movie, rank);  // ✅ rank 파라미터 추가
    moviesList.appendChild(movieCard);
    
    // 결과 영역 표시
    searchResults.style.display = 'block';
    
    // 새 영화에 애니메이션 효과
    movieCard.style.opacity = '0';
    movieCard.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        movieCard.style.transition = 'all 0.3s ease';
        movieCard.style.opacity = '1';
        movieCard.style.transform = 'translateY(0)';
    }, 10);
}

// 빠른 검색 (사전 로딩된 모델 활용)
async function handleFastSearch(query) {
    console.log('🚀 [DEBUG] handleFastSearch 시작, 검색어:', query);
    
    try {
        console.log('🌐 [DEBUG] /fast-search로 POST 요청 전송 중...');
        
        const response = await fetch('/fast-search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        console.log('📡 [DEBUG] 서버 응답 상태:', response.status, response.statusText);
        
        const data = await response.json();
        console.log('📦 [DEBUG] 서버 응답 데이터:', data);
        
        if (!response.ok) {
            throw new Error(data.error || '빠른 검색 실패');
        }
        
        currentSearchResults = data;
        showResults(data);
        
    } catch (error) {
        console.error('❌ [DEBUG] 빠른 검색 오류:', error);
        throw error; // 상위로 에러 전파
    }
}

// 🚀 실시간 LLM 스트리밍 검색 (진짜 버전)
async function handleStreamingSearch(query) {
    console.log('🎬 실시간 LLM 스트리밍 검색 시작:', query);
    
    // 결과 초기화
    streamingResults = [];
    showResults('🔍 LLM 실시간 분석 시작...', []);
    
    return new Promise((resolve, reject) => {
        try {
            // 진짜 스트리밍 검색 요청
            fetch('/streaming-search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            }).then(response => {
                if (!response.ok) {
                    throw new Error(`서버 오류: ${response.status}`);
                }
                
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                
                function readStream() {
                    reader.read().then(({ done, value }) => {
                        if (done) {
                            console.log('🎉 스트리밍 완료');
                            // 스트리밍 완료 시 currentSearchResults 설정
                            currentSearchResults = {
                                query: query,
                                movies: streamingResults,
                                total_count: streamingResults.length,
                                llm_filtered: true
                            };
                            resolve();
                            return;
                        }
                        
                        const chunk = decoder.decode(value);
                        buffer += chunk;
                        
                        const lines = buffer.split('\n');
                        buffer = lines.pop() || ''; // 마지막 불완전한 라인은 버퍼에 보관
                        
                        lines.forEach(line => {
                            if (line.startsWith('data: ')) {
                                try {
                                    const jsonStr = line.substring(6).trim();
                                    if (jsonStr) {
                                        const data = JSON.parse(jsonStr);
                                        handleStreamingMessage(data);
                                    }
                                } catch (error) {
                                    console.error('스트리밍 데이터 파싱 오류:', error, 'Line:', line);
                                }
                            }
                        });
                        
                        readStream();
                    }).catch(error => {
                        console.error('스트리밍 읽기 오류:', error);
                        reject(error);
                    });
                }
                
                readStream();
            }).catch(error => {
                console.error('스트리밍 요청 오류:', error);
                reject(error);
            });
            
        } catch (error) {
            console.error('스트리밍 검색 오류:', error);
            reject(error);
        }
    });
}

// 일반 검색 (폴백)
async function handleRegularSearch(query) {
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

// 스트리밍 결과 업데이트
function updateStreamingResults(query) {
    const data = {
        query: query,
        movies: streamingResults,
        total_count: streamingResults.length,
        llm_filtered: true
    };
    
    currentSearchResults = data;
    showResults(data);
}

// 결과 카운트 업데이트 (실시간)
function updateResultsCount(count, completed = false) {
    const resultsCount = document.getElementById('results-count');
    const resultsTitle = document.getElementById('results-title');
    
    if (resultsCount) {
        resultsCount.textContent = count;
    }
    
    if (resultsTitle) {
        if (completed) {
            resultsTitle.textContent = `검색 완료 (${count}개 결과)`;
        } else {
            resultsTitle.textContent = `실시간 검색 중... (${count}개 발견)`;
        }
    }
}

// 로딩 상태 표시
function showLoading() {
    hideAllStates();
    loadingState.classList.remove('hidden');
}

// 로딩 상태 숨기기
function hideLoading() {
    loadingState.classList.add('hidden');
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
    
    // 줄거리 축약 (100자로 제한)
    const plot = movie.plot || '줄거리 정보가 없습니다.';
    const shortPlot = plot.length > 100 ? plot.substring(0, 100) + '...' : plot;
    
    // 포스터 이미지 경로 설정 (백엔드에서 poster_url로 보내줌)
    const posterUrl = movie.poster_url || movie.poster;
    const posterHTML = posterUrl ? 
        `<img src="${posterUrl}" alt="${movie.title}" onerror="this.style.display='none'; this.parentElement.classList.add('no-poster');" />` :
        `<div class="poster-fallback"><i class="fas fa-film"></i></div>`;
    
    card.innerHTML = `
        <div class="movie-poster" onclick="goToMovieDetail(${rank})">
            ${posterHTML}
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

// 영화 상세 페이지로 이동 (제목과 연도 기반)
function goToMovieDetail(rank) {
    // 어떤 결과 배열을 사용할지 결정 (스트리밍 우선)
    const resultsArray = 
        (currentSearchResults && currentSearchResults.movies) || 
        streamingResults;
    
    if (!resultsArray || resultsArray.length < rank) {
        console.error('영화 데이터를 찾지 못했습니다.', {
            currentSearchResults,
            streamingResults,
            rank
        });
        return;
    }
    
    const movie = resultsArray[rank - 1];
    const title = encodeURIComponent(movie.title || 'Unknown');
    const year = encodeURIComponent(movie.year || 'Unknown');
    
    // 제목과 연도 기반 URL로 이동
    window.location.href = `/movie/${title}/${year}`;
}

// LLM 설명 모달 표시 (수정: JSON 데이터에서 직접 가져오기)
async function showExplanation(rank) {
    // 모달 열기
    explanationModal.classList.remove('hidden');
    modalLoading.classList.remove('hidden');
    modalExplanation.classList.add('hidden');
    
    // 어떤 결과 배열을 사용할지 결정 (스트리밍 우선)
    const resultsArray = 
        (currentSearchResults && currentSearchResults.movies) || 
        streamingResults;
    
    if (!resultsArray || resultsArray.length < rank) {
        console.error('설명에 사용할 영화 데이터를 찾지 못했습니다.', {
            currentSearchResults,
            streamingResults,
            rank
        });
        modalTitle.textContent = '설명 오류';
        showModalExplanation('해당 영화의 설명 데이터를 찾을 수 없습니다.');
        return;
    }
    
    // 영화 정보 가져오기
    const movie = resultsArray[rank - 1];
    modalTitle.textContent = `"${movie.title || 'Unknown'}" 추천 이유`;
    
    // 검색 결과 JSON에서 직접 설명 가져오기
    let explanation = '설명 정보가 없습니다.';
    
    if (movie.llm_analysis && movie.llm_analysis.reason) {
        explanation = movie.llm_analysis.reason;
    } else {
        console.warn('LLM 분석 데이터가 없습니다:', movie);
        explanation = `${movie.title} 영화에 대한 LLM 분석 정보가 누락되었습니다.`;
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

// 🔒 검색 상태 관리 함수들
function saveSearchState() {
    try {
        const state = {
            query: searchInput.value.trim(),
            currentSearchResults,
            streamingResults,
            scrollTop: window.scrollY
        };
        sessionStorage.setItem(SEARCH_STATE_KEY, JSON.stringify(state));
    } catch (e) {
        console.warn('검색 상태 저장 실패:', e);
    }
}

// 🔓 세션 스토리지에서 검색 상태 복원
function restoreSearchState() {
    try {
        const raw = sessionStorage.getItem(SEARCH_STATE_KEY);
        if (!raw) return;

        const state = JSON.parse(raw);

        // 검색창 복원
        if (state.query) {
            searchInput.value = state.query;
        }

        currentSearchResults = state.currentSearchResults || null;
        streamingResults = state.streamingResults || (currentSearchResults ? currentSearchResults.movies : []);

        // 화면에 다시 그림
        if (currentSearchResults && currentSearchResults.movies) {
            showResults(currentSearchResults);
        } else if (streamingResults && streamingResults.length > 0) {
            showResults({
                query: state.query || '',
                movies: streamingResults
            });
        }

        // 스크롤 위치 복원
        if (typeof state.scrollTop === 'number') {
            setTimeout(() => {
                window.scrollTo(0, state.scrollTop);
            }, 50);
        }
    } catch (e) {
        console.warn('검색 상태 복원 실패:', e);
    }
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

// 페이지 로드 시 검색 상태 복원
window.addEventListener('load', function() {
    // 이전 검색 상태 복원 (있으면)
    restoreSearchState();

    // 아무 상태도 없으면 포커스만
    if (!sessionStorage.getItem(SEARCH_STATE_KEY)) {
        searchInput.focus();
    }
});