"""
GPT API 분석 결과 시각화
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import os

class GPTVisualizationAnalyzer:
    def __init__(self):
        """
        GPT 분석 결과 시각화 초기화
        """
        # 한글 폰트 설정
        plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 결과 저장 디렉토리
        self.output_dir = "gpt_visualization_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 {self.output_dir} 디렉토리 생성")
    
    def load_gpt_results(self, file_path):
        """
        GPT 분석 결과 로드
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"❌ {file_path} 파일을 찾을 수 없습니다.")
            return None
        except json.JSONDecodeError:
            print(f"❌ {file_path} JSON 파싱 오류")
            return None
    
    def create_flow_curve_chart(self, movie_title, data):
        """
        스토리 흐름 곡선 차트 생성 (설명 제외)
        """
        flow_curve = data['flow_curve']
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 흐름 곡선 그리기
        segments = range(1, 11)
        line = ax.plot(segments, flow_curve, 'b-', linewidth=3, marker='o', 
                      markersize=8, label='스토리 흐름 곡선')
        
        # 각 구간에 점수 표시
        for i, (segment, intensity) in enumerate(zip(segments, flow_curve)):
            ax.annotate(f'{intensity}', (segment, intensity), 
                       textcoords="offset points", xytext=(0,10), ha='center',
                       fontsize=12, fontweight='bold', color='blue')
        
        # 차트 꾸미기
        ax.set_title(f'{movie_title} - 스토리 흐름 곡선', fontsize=16, fontweight='bold')
        ax.set_xlabel('영화 구간 (1-10)', fontsize=12)
        ax.set_ylabel('긴장감/강도 (1-10)', fontsize=12)
        ax.set_xticks(segments)
        ax.set_ylim(0, 11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 배경색과 스타일
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # 저장
        filename = f"{self.output_dir}/{movie_title.replace(' ', '_')}_gpt_flow.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ {movie_title} 흐름 곡선 저장: {filename}")
        plt.show()
        return filename
    
    def create_flow_analysis_text(self, movie_title, data):
        """
        구간별 상세 분석 텍스트 파일 생성
        """
        detailed_analysis = data.get('detailed_analysis', [])
        flow_curve = data['flow_curve']
        
        if not detailed_analysis:
            return None
            
        # 텍스트 파일 생성
        filename = f"{self.output_dir}/{movie_title.replace(' ', '_')}_gpt_analysis.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"🎬 {movie_title} - GPT 구간별 상세 분석\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"흐름 곡선: {flow_curve}\n\n")
            
            for i, analysis in enumerate(detailed_analysis, 1):
                intensity = flow_curve[i-1] if i-1 < len(flow_curve) else "N/A"
                f.write(f"구간 {i} (강도: {intensity}/10):\n")
                f.write(f"{analysis}\n\n")
        
        print(f"✅ {movie_title} 상세 분석 저장: {filename}")
        return filename
    
    def create_genre_chart(self, movie_title, data):
        """
        장르 분석 막대 그래프 생성 (모든 영화 동일한 장르 순서)
        """
        genres = data['genres']
        
        # 고정된 장르 순서 (모든 영화에서 동일)
        fixed_genre_order = ['action', 'comedy', 'drama', 'horror', 'romance', 'sci_fi', 'thriller']
        
        # 고정 순서에 맞춰 점수 정리 (없는 장르는 0점)
        genre_names = []
        genre_scores = []
        for genre in fixed_genre_order:
            genre_names.append(genre.replace('_', '-').title())
            genre_scores.append(genres.get(genre, 0))
        
        # 색상 설정 (점수에 따라)
        colors = ['#ff6b6b' if score >= 8 else '#4ecdc4' if score >= 5 else '#95a5a6' 
                 for score in genre_scores]
        
        # 그래프 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.bar(genre_names, genre_scores, color=colors, alpha=0.8, edgecolor='black')
        
        # 막대 위에 점수 표시
        for bar, score in zip(bars, genre_scores):
            if score > 0:  # 0점인 경우는 표시하지 않음
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{score}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # 차트 꾸미기
        ax.set_title(f'{movie_title} - 장르 분석', fontsize=16, fontweight='bold')
        ax.set_xlabel('장르', fontsize=12)
        ax.set_ylabel('점수 (1-10)', fontsize=12)
        ax.set_ylim(0, 11)
        
        # x축 레이블 회전
        plt.xticks(rotation=45, ha='right')
        
        # 격자 추가
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # 범례 추가
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#ff6b6b', label='높음 (8-10)'),
                          Patch(facecolor='#4ecdc4', label='보통 (5-7)'),
                          Patch(facecolor='#95a5a6', label='낮음 (1-4)')]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 저장
        filename = f"{self.output_dir}/{movie_title.replace(' ', '_')}_gpt_genres.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ {movie_title} 장르 분석 저장: {filename}")
        plt.show()
        return filename
    
    def create_comparison_chart(self, results):
        """
        영화들 간 비교 차트 생성
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        movie_titles = list(results.keys())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        # 1. 흐름 곡선 비교
        for i, (title, data) in enumerate(results.items()):
            segments = range(1, 11)
            flow_curve = data['flow_curve']
            ax1.plot(segments, flow_curve, 'o-', linewidth=2.5, markersize=6, 
                    label=title, color=colors[i % len(colors)])
        
        ax1.set_title('🎬 GPT 분석 - 영화별 스토리 흐름 비교', fontsize=14, fontweight='bold')
        ax1.set_xlabel('영화 구간')
        ax1.set_ylabel('긴장감/강도')
        ax1.set_xticks(range(1, 11))
        ax1.set_ylim(0, 11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_facecolor('#f8f9fa')
        
        # 2. 주요 장르 비교 (고정된 순서)
        fixed_genre_order = ['action', 'comedy', 'drama', 'horror', 'romance', 'sci_fi', 'thriller']
        genre_list = [genre.replace('_', '-').title() for genre in fixed_genre_order]
        
        x = np.arange(len(genre_list))
        width = 0.35
        
        for i, (title, data) in enumerate(results.items()):
            genre_scores = [data['genres'].get(genre, 0) for genre in fixed_genre_order]
            offset = (i - len(results)/2 + 0.5) * width
            bars = ax2.bar(x + offset, genre_scores, width, label=title, 
                          color=colors[i % len(colors)], alpha=0.8)
            
            # 막대 위에 점수 표시
            for bar, score in zip(bars, genre_scores):
                if score > 0:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                           f'{score}', ha='center', va='bottom', fontsize=9)
        
        ax2.set_title('🎭 GPT 분석 - 영화별 장르 비교', fontsize=14, fontweight='bold')
        ax2.set_xlabel('장르')
        ax2.set_ylabel('점수')
        ax2.set_xticks(x)
        ax2.set_xticklabels(genre_list, rotation=45, ha='right')
        ax2.set_ylim(0, 11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        ax2.set_facecolor('#f8f9fa')
        
        # 저장
        filename = f"{self.output_dir}/gpt_movies_comparison.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ 영화 비교 차트 저장: {filename}")
        plt.show()
        return filename
    
    def visualize_all(self, results_file):
        """
        모든 시각화 생성
        """
        print("🎨 GPT 분석 결과 시각화 시작!")
        print("=" * 50)
        
        # 결과 로드
        results = self.load_gpt_results(results_file)
        if not results:
            return
        
        print(f" {len(results)}개 영화 데이터 로드 완료")
        
        # 각 영화별 개별 차트 생성
        for movie_title, data in results.items():
            print(f"\n🎬 {movie_title} 시각화 중...")
            self.create_flow_curve_chart(movie_title, data)
            self.create_flow_analysis_text(movie_title, data)
            self.create_genre_chart(movie_title, data)
        
        # 비교 차트 생성
        print(f"\n 영화 비교 차트 생성 중...")
        self.create_comparison_chart(results)
        
        print(f"\n 모든 시각화 완료! 결과는 '{self.output_dir}' 폴더에 저장되었습니다.")
        return self.output_dir

def main():
    """
    메인 실행 함수
    """
    visualizer = GPTVisualizationAnalyzer()
    results_dir = visualizer.visualize_all('gpt_api_analysis_results.json')
    
    print(f"\n 결과 확인 방법:")
    print(f"1. 파일 탐색기에서 '{results_dir}' 폴더 열기")
    print(f"2. 생성된 PNG 파일들 확인")

if __name__ == "__main__":
    main()
