"""
beanllm Admin Dashboard - Streamlit
Google Workspace 모니터링 및 관리 대시보드

실행 방법:
    streamlit run admin/dashboard.py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# beanllm 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from beanllm.facade.core.client_facade import Client
    from beanllm.infrastructure.distributed.google_events import (
        get_google_export_stats,
        get_security_events,
        log_admin_action,
    )

    BEANLLM_AVAILABLE = True
except ImportError:
    BEANLLM_AVAILABLE = False

# 페이지 설정
st.set_page_config(
    page_title="beanllm Admin Dashboard",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 커스텀 CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #10b981;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .status-good {
        color: #10b981;
        font-weight: 600;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    .status-critical {
        color: #ef4444;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)


def check_dependencies() -> bool:
    """필수 의존성 확인"""
    if not BEANLLM_AVAILABLE:
        st.error("❌ beanllm not available. Please install beanllm.")
        return False

    # MongoDB URI 확인
    if not os.getenv("MONGODB_URI"):
        st.warning("⚠️ MONGODB_URI not set. Statistics features will not work.")
        st.info("Set MONGODB_URI in your environment: `export MONGODB_URI='mongodb+srv://...'`")
        return False

    # Gemini API 키 확인 (선택적)
    if not os.getenv("GEMINI_API_KEY"):
        st.info("ℹ️ GEMINI_API_KEY not set. AI analysis features will be disabled.")
        st.info("Set GEMINI_API_KEY to enable Gemini-powered insights.")

    return True


def render_header():
    """대시보드 헤더"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            '<div class="main-header">👑 beanllm Admin Dashboard</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="sub-header">Google Workspace Monitoring & Analytics</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(f"**Last Updated**: {datetime.now().strftime('%H:%M:%S')}")
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()


async def get_stats_async(hours: int) -> Dict[str, Any]:
    """비동기로 통계 가져오기"""
    return await get_google_export_stats(hours=hours)


async def get_security_async(hours: int, severity: str = "high") -> List[Dict[str, Any]]:
    """비동기로 보안 이벤트 가져오기"""
    return await get_security_events(hours=hours, severity=severity)


async def analyze_with_gemini_async(stats: Dict[str, Any], hours: int) -> str:
    """Gemini로 분석"""
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        return "Gemini API key not configured."

    prompt = f"""
다음은 지난 {hours}시간 동안의 Google Workspace 사용 통계입니다:

총 내보내기: {stats['total_exports']}
서비스별: {json.dumps(stats['by_service'], ensure_ascii=False)}
상위 사용자: {stats['top_users'][:5]}

간결한 분석 (3-5문장):
1. 주요 사용 패턴
2. 이상 징후
3. 권장 조치
"""

    try:
        client = Client(model="gemini-2.0-flash-exp", provider="google")
        response = await client.chat([{"role": "user", "content": prompt}])
        return response.content
    except Exception as e:
        return f"Gemini analysis failed: {str(e)}"


def render_overview_tab():
    """개요 탭"""
    st.header("📊 실시간 통계")

    # 시간 범위 선택
    hours = st.selectbox(
        "조회 기간",
        options=[1, 6, 12, 24, 72, 168],
        index=3,
        format_func=lambda x: f"최근 {x}시간" if x < 24 else f"최근 {x//24}일",
    )

    # 통계 로드
    try:
        with st.spinner("Loading statistics..."):
            stats = asyncio.run(get_stats_async(hours))

        # 메트릭 카드
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="📤 Total Exports",
                value=stats["total_exports"],
                delta=f"+{stats['total_exports'] - stats.get('prev_total', 0)}"
                if stats.get("prev_total")
                else None,
            )

        with col2:
            st.metric(
                label="👥 Active Users",
                value=len(stats["top_users"]),
            )

        with col3:
            most_popular = (
                max(stats["by_service"], key=stats["by_service"].get)
                if stats["by_service"]
                else "N/A"
            )
            st.metric(
                label="🔥 Most Popular",
                value=most_popular.capitalize() if most_popular != "N/A" else "N/A",
            )

        with col4:
            avg_per_user = (
                stats["total_exports"] / len(stats["top_users"]) if stats["top_users"] else 0
            )
            st.metric(label="📊 Avg/User", value=f"{avg_per_user:.1f}")

        # 서비스별 사용량 차트
        st.subheader("서비스별 사용량")

        if stats["by_service"]:
            import pandas as pd

            df = pd.DataFrame(
                [
                    {"Service": k.capitalize(), "Count": v}
                    for k, v in sorted(
                        stats["by_service"].items(), key=lambda x: x[1], reverse=True
                    )
                ]
            )

            st.bar_chart(df.set_index("Service"))

        # 상위 사용자
        st.subheader("👥 상위 사용자")

        if stats["top_users"]:
            user_data = []
            for i, (user_id, count) in enumerate(stats["top_users"][:10], 1):
                percentage = (
                    (count / stats["total_exports"] * 100) if stats["total_exports"] > 0 else 0
                )
                user_data.append(
                    {
                        "Rank": i,
                        "User ID": user_id,
                        "Exports": count,
                        "Percentage": f"{percentage:.1f}%",
                    }
                )

            st.dataframe(user_data, use_container_width=True)
        else:
            st.info("No user data available")

    except Exception as e:
        st.error(f"Failed to load statistics: {e}")


def render_analysis_tab():
    """AI 분석 탭"""
    st.header("🤖 Gemini AI Analysis")

    # Gemini API 키 확인
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        st.warning("⚠️ Gemini API key not configured")
        st.info("Set GEMINI_API_KEY in your environment to enable AI-powered analysis.")
        return

    hours = st.slider("분석 기간 (시간)", min_value=1, max_value=168, value=24)

    if st.button("🚀 Analyze with Gemini", use_container_width=True, type="primary"):
        try:
            with st.spinner("Loading statistics..."):
                stats = asyncio.run(get_stats_async(hours))

            with st.spinner("🤖 Analyzing with Gemini... (This may take a few seconds)"):
                analysis = asyncio.run(analyze_with_gemini_async(stats, hours))

            # 분석 결과 표시
            st.success("✅ Analysis complete!")

            st.markdown("### 📊 Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Exports", stats["total_exports"])
            with col2:
                st.metric("Active Users", len(stats["top_users"]))
            with col3:
                most_popular = (
                    max(stats["by_service"], key=stats["by_service"].get)
                    if stats["by_service"]
                    else "N/A"
                )
                st.metric(
                    "Most Popular", most_popular.capitalize() if most_popular != "N/A" else "N/A"
                )

            st.markdown("### 🧠 Gemini Insights")
            st.info(analysis)

            # 관리자 액션 로깅
            try:
                asyncio.run(
                    log_admin_action(
                        admin_id=st.session_state.get("admin_id", "admin"),
                        action="gemini_analysis",
                        metadata={"hours": hours, "total_exports": stats["total_exports"]},
                    )
                )
            except:
                pass

        except Exception as e:
            st.error(f"Analysis failed: {e}")


def render_security_tab():
    """보안 탭"""
    st.header("🔒 Security Events")

    # 설정
    col1, col2 = st.columns(2)
    with col1:
        hours = st.selectbox(
            "조회 기간",
            options=[1, 6, 12, 24, 72, 168],
            index=3,
            format_func=lambda x: f"최근 {x}시간" if x < 24 else f"최근 {x//24}일",
        )
    with col2:
        severity = st.selectbox("심각도", options=["low", "medium", "high"], index=2)

    # 보안 이벤트 로드
    try:
        with st.spinner("Loading security events..."):
            events = asyncio.run(get_security_async(hours, severity))

        if not events:
            st.success(f"✅ No {severity}-severity security events in the last {hours} hours")
            st.balloons()
            return

        # 경고 메시지
        st.warning(f"⚠️ Found {len(events)} {severity}-severity events")

        # 이벤트 테이블
        event_data = []
        for event in events[:50]:  # 최근 50개만
            event_data.append(
                {
                    "Time": event.get("timestamp", ""),
                    "User": event.get("user_id", "unknown"),
                    "Reason": event.get("reason", ""),
                    "Severity": event.get("severity", "").upper(),
                }
            )

        st.dataframe(event_data, use_container_width=True)

        # Gemini 분석 버튼
        if os.getenv("GEMINI_API_KEY") and st.button(
            "🤖 Analyze with Gemini", use_container_width=True
        ):
            with st.spinner("🤖 Analyzing security events with Gemini..."):
                prompt = f"""
보안 이벤트 분석:
{json.dumps(events[:5], indent=2, ensure_ascii=False)}

다음을 제공해주세요:
1. 위협 수준 (Low/Medium/High/Critical)
2. 즉시 대응 필요 사항
3. 예방 조치
"""
                try:
                    client = Client(model="gemini-2.0-flash-exp", provider="google")
                    response = await client.chat([{"role": "user", "content": prompt}])
                    st.error("🛡️ Security Analysis")
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

    except Exception as e:
        st.error(f"Failed to load security events: {e}")


def render_cost_tab():
    """비용 최적화 탭"""
    st.header("💰 Cost Optimization")

    st.info("""
    **현재 구성**:
    - MongoDB Atlas: Free tier (512MB)
    - Upstash Redis: Free tier (10K commands/day)
    - Gemini API: User's paid key (no additional cost)
    """)

    if st.button("🤖 Get Optimization Recommendations", use_container_width=True, type="primary"):
        try:
            # 통계 로드
            with st.spinner("Loading usage data..."):
                stats_24h = asyncio.run(get_stats_async(24))
                stats_7d = asyncio.run(get_stats_async(168))

            with st.spinner("🤖 Generating recommendations with Gemini..."):
                prompt = f"""
비용 최적화 분석:

사용량:
- 24시간: {stats_24h['total_exports']} exports
- 7일: {stats_7d['total_exports']} exports

현재 구성:
- MongoDB Atlas Free (512MB)
- Upstash Redis Free (10K commands/day)

다음을 제공해주세요:
1. 무료 티어 초과 위험 평가
2. 비용 절감 방안 (구체적)
3. 예상 월간 비용 ($0-$20 범위)
4. 즉시 실행 가능한 조치

간결하고 실용적인 조언을 제공해주세요.
"""

                client = Client(model="gemini-2.0-flash-exp", provider="google")
                response = await client.chat([{"role": "user", "content": prompt}])

                st.success("✅ Analysis complete!")
                st.markdown("### 💡 Recommendations")
                st.info(response.content)

        except Exception as e:
            st.error(f"Optimization analysis failed: {e}")


def render_settings_tab():
    """설정 탭"""
    st.header("⚙️ Settings")

    st.subheader("환경 변수")

    # 환경 변수 확인
    env_vars = {
        "MONGODB_URI": os.getenv("MONGODB_URI", "Not set"),
        "GEMINI_API_KEY": "Set" if os.getenv("GEMINI_API_KEY") else "Not set",
        "GOOGLE_API_KEY": "Set" if os.getenv("GOOGLE_API_KEY") else "Not set",
    }

    for key, value in env_vars.items():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.text(key)
        with col2:
            if value == "Not set":
                st.error(f"❌ {value}")
            else:
                st.success(f"✅ {value[:20]}..." if value.startswith("mongodb") else f"✅ {value}")

    st.divider()

    st.subheader("관리자 정보")

    # 세션 상태로 관리자 ID 관리
    if "admin_id" not in st.session_state:
        st.session_state.admin_id = "admin"

    admin_id = st.text_input("Admin ID", value=st.session_state.admin_id)
    if st.button("Update Admin ID"):
        st.session_state.admin_id = admin_id
        st.success(f"✅ Admin ID updated to: {admin_id}")

    st.divider()

    st.subheader("대시보드 정보")

    st.markdown("""
    **beanllm Admin Dashboard**
    - Version: 1.0.0
    - Framework: Streamlit
    - Features: Real-time monitoring, AI-powered insights, Security alerts

    **Contact**: admin@beanllm.com
    """)


def main():
    """메인 대시보드"""

    # 의존성 확인
    if not check_dependencies():
        st.stop()

    # 헤더
    render_header()

    # 사이드바
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")

        tabs = st.radio(
            "Select a tab:",
            options=["Overview", "AI Analysis", "Security", "Cost", "Settings"],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown("### 📊 Quick Stats")
        try:
            stats = asyncio.run(get_stats_async(24))
            st.metric("24h Exports", stats["total_exports"])
            st.metric("Active Users", len(stats["top_users"]))
        except:
            st.info("Stats unavailable")

        st.divider()

        st.markdown("### 🔗 Quick Links")
        st.markdown("- [beanllm Docs](https://github.com)")
        st.markdown("- [Report Issue](https://github.com/issues)")

    # 탭 렌더링
    if tabs == "Overview":
        render_overview_tab()
    elif tabs == "AI Analysis":
        render_analysis_tab()
    elif tabs == "Security":
        render_security_tab()
    elif tabs == "Cost":
        render_cost_tab()
    elif tabs == "Settings":
        render_settings_tab()


if __name__ == "__main__":
    main()
