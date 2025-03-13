import Container from "@/app/_components/container";
import { Intro } from "@/app/_components/intro";
import { Stories } from "@/app/_components/stories";
import { getAllPosts } from "@/lib/api";

export default function Index() {
  const allPosts = getAllPosts();

  return (
    <main>
      <Container>
        <Intro />
        <Stories posts={allPosts} />
      </Container>
    </main>
  );
}
