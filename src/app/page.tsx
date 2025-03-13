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
        <PostPreviewList posts={allPosts} />
      </Container>
    </main>
  );
}
