import Container from "@/app/_components/container";
import { Intro } from "@/app/_components/intro";
import { PostPreviewList } from "@/app/_components/post-preview-list";
import { getAllPosts } from "@/lib/api";

export default function Index() {
  const allPosts = getAllPosts();

  return (
    <main>
      <Container>
        <Intro />
        <h2 className="mb-8 text-5xl md:text-7xl font-bold tracking-tighter leading-tight">
          Thoughts & more
        </h2>
        <PostPreviewList posts={allPosts} />
      </Container>
    </main>
  );
}
