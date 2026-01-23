"use client";

import Link from "next/link";
import { PageLayout } from "@/components/PageLayout";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { BeanIcon } from "@/components/icons/BeanIcon";
import {
  MessageSquare,
  FileText,
  Bot,
  Users,
  Network,
  Image,
  Music,
  BarChart3,
  Settings,
  Search,
  ArrowRight,
  Sparkles,
} from "lucide-react";

const features = [
  {
    icon: MessageSquare,
    name: "Chat",
    description: "일반 대화",
  },
  {
    icon: FileText,
    name: "RAG",
    description: "문서 검색 & 질의응답",
  },
  {
    icon: Users,
    name: "Multi-Agent",
    description: "에이전트 협업",
  },
  {
    icon: Network,
    name: "Knowledge Graph",
    description: "지식 그래프 구축 & 탐색",
  },
  {
    icon: Music,
    name: "Audio",
    description: "음성 인식 & 전사",
  },
  {
    icon: Image,
    name: "OCR",
    description: "이미지 텍스트 인식",
  },
  {
    icon: FileText,
    name: "Google Workspace",
    description: "Docs, Drive, Gmail 연동",
  },
  {
    icon: Search,
    name: "Web Search",
    description: "웹 검색",
  },
];

export default function HomePage() {
  return (
    <PageLayout
      title="BeanLLM Playground"
      description="Production-Ready LLM Toolkit with Clean Architecture"
    >
      <div className="space-y-8">
        {/* Hero Section */}
        <Card className="bg-gradient-to-br from-primary/5 via-primary/10 to-transparent border-primary/20">
          <CardHeader className="text-center space-y-4 pb-8">
            <div className="flex justify-center">
              <div className="w-16 h-16 flex items-center justify-center text-primary">
                <BeanIcon className="w-full h-full" />
              </div>
            </div>
            <div>
              <CardTitle className="text-3xl mb-2">
                Unified Chat Interface
              </CardTitle>
              <CardDescription className="text-base">
                모든 기능을 하나의 채팅 인터페이스에서 사용하세요
              </CardDescription>
            </div>
            <div className="flex justify-center pt-2">
              <Link href="/chat">
                <Button size="lg" className="gap-2">
                  <Sparkles className="h-4 w-4" />
                  Start Chatting
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
            </div>
          </CardHeader>
        </Card>

        {/* Features Grid */}
        <div>
          <h2 className="text-2xl font-semibold mb-4 text-center">
            Available Features
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {features.map((feature) => {
              const Icon = feature.icon;
              return (
                <Card
                  key={feature.name}
                  className="hover:bg-accent transition-colors"
                >
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <Icon className="h-5 w-5 text-primary" />
                      <CardTitle className="text-base">
                        {feature.name}
                      </CardTitle>
                    </div>
                    <CardDescription className="text-sm">
                      {feature.description}
                    </CardDescription>
                  </CardHeader>
                </Card>
              );
            })}
          </div>
        </div>

        {/* Tech Stack */}
        <Card>
          <CardHeader>
            <CardTitle>Technology Stack</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="font-medium mb-1">LLM Providers</p>
                <p className="text-muted-foreground text-xs">
                  OpenAI, Anthropic, Google, Ollama
                </p>
              </div>
              <div>
                <p className="font-medium mb-1">Vector Stores</p>
                <p className="text-muted-foreground text-xs">
                  Chroma, FAISS, Qdrant, Pinecone
                </p>
              </div>
              <div>
                <p className="font-medium mb-1">Infrastructure</p>
                <p className="text-muted-foreground text-xs">
                  Redis, Kafka, MongoDB
                </p>
              </div>
              <div>
                <p className="font-medium mb-1">Framework</p>
                <p className="text-muted-foreground text-xs">
                  Clean Architecture, FastAPI, Next.js 15
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </PageLayout>
  );
}
